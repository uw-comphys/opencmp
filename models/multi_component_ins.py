########################################################################################################################
# Copyright 2021 the authors (see AUTHORS file for full list).                                                         #
#                                                                                                                      #
# This file is part of OpenCMP.                                                                                        #
#                                                                                                                      #
# OpenCMP is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public  #
# License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any     #
# later version.                                                                                                       #
#                                                                                                                      #
# OpenCMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied        #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more  #
# details.                                                                                                             #
#                                                                                                                      #
# You should have received a copy of the GNU Lesser General Public License along with OpenCMP. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

import ngsolve as ngs

from helpers.ngsolve_ import get_special_functions
from helpers.dg import avg, jump, grad_avg
from models import Model
from config_functions import ConfigParser
from typing import Dict, List, Optional, Union, cast
from ngsolve import BilinearForm, FESpace, Grad, InnerProduct, LinearForm, Norm, GridFunction, OuterProduct, div, dx, \
    Parameter, HDiv, Preconditioner, IfPos
from ngsolve.comp import ProxyFunction
from helpers.error import norm, mean


# TODO: These might be a good reference
# https://github.com/NGSolve/modeltemplates/tree/master/templates
# https://github.com/NGSolve/modeltemplates/blob/master/introduction/introduction.ipynb


class MultiComponentINS(Model):
    """
    A single phase multicomponent incompressible Navier-Stokes model.
    """

    def __init__(self, config: ConfigParser, t_param: Parameter) -> None:
        # Specify information about the model components
        # NOTE: These MUST be set before calling super(), since it's needed in superclass' __init__
        self.model_components               = {'u': 0,      'p': 1}
        self.model_local_error_components   = {'u': True,   'p': False}
        self.time_derivative_components     = {'u': True,   'p': False}

        # Read in and add the multiple components
        self._add_multiple_components(config)

        # Remove the entries associated with velocity and pressure, so as to have only the extra components left
        # NOTE: ugly hack here since I need extra_components_inverted before calling super(),
        #       but model_components_inverted is only defined inside super().
        self.extra_components_inverted = dict(zip(self.model_components.values(), self.model_components.keys()))
        self.extra_components_inverted.pop(0)
        self.extra_components_inverted.pop(1)

        # Pre-define which BCs are accepted for this model, all others are thrown out.
        # TODO: For James. Neumann BC or diffusive flux BC? Which makes more sense?
        # TODO: If we go with a Neumann BC need to add in the diffusion coefficient in the bilinear form.
        self.BC_init = {'dirichlet': {},
                        'neumann': {},
                        'stress': {},
                        'pinned': {},
                        'total_flux': {},
                        'surface_rxn': {}}

        super().__init__(config, t_param)

        if self.DIM or self.DG:
            raise NotImplementedError('DIM and DG is not yet implemented.')

        # Load the solver parameters.
        self.linearize = self.config.get_item(['SOLVER', 'linearization_method'], str)
        if self.linearize not in ['Oseen', 'IMEX']:
            raise TypeError('Don\'t recognize the linearization method.')

        if self.linearize == 'Oseen':
            nonlinear_tolerance: Dict[str, float] = self.config.get_dict(['SOLVER', 'nonlinear_tolerance'], None)
            self.abs_nonlinear_tolerance = nonlinear_tolerance['absolute']
            self.rel_nonlinear_tolerance = nonlinear_tolerance['relative']
            self.nonlinear_max_iters = self.config.get_item(['SOLVER', 'nonlinear_max_iterations'], int)
            if self.nonlinear_max_iters < 1:
                raise ValueError('Nonlinear solve must involve at least one iteration. Set nonlinear_max_iterations to '
                                 'at least 1.')
            self.W = self._construct_linearization_terms()

    @staticmethod
    def allows_explicit_schemes() -> bool:
        # MultiComponentINS cannot work with explicit schemes
        return False

    def _set_model_parameters(self) -> None:
        self.kv = self.model_functions.model_parameters_dict['kinematic_viscosity']['all']
        self.f = self.model_functions.model_functions_dict['source']
        self.Ds = self.model_functions.model_parameters_dict['diffusion_coefficients']

        # Ensure that a value was loaded for each extra component
        assert len(self.Ds) == len(self.config.get_list(['OTHER', 'component_names'], str))

    def _construct_fes(self) -> FESpace:
        if not self.DG:
            if self.element['u'] == 'HDiv' or self.element['u'] == 'RT':
                print('We recommended that you NOT use HDIV spaces without DG due to numerical issues.')
            if self.element['p'] == 'L2':
                print('We recommended that you NOT use L2 spaces without DG due to numerical issues.')

        if self.element['u'] == 'RT':
            # Raviart-Thomas elements are a type of HDiv finite element.
            fes_u = HDiv(self.mesh, order=self.interp_ord, dirichlet=self.dirichlet_names.get('u', ''),
                         dgjumps=self.DG, RT=True)
        else:
            fes_u = getattr(ngs, self.element['u'])(self.mesh, order=self.interp_ord,
                                                    dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)

        if self.element['p'] == 'L2' and 'p' in self.dirichlet_names.keys():
            raise ValueError('Not able to pin pressure at a point on L2 spaces.')
        else:
            fes_p = getattr(ngs, self.element['p'])(self.mesh, order=self.interp_ord - 1,
                                                    dirichlet=self.dirichlet_names.get('p', ''), dgjumps=self.DG)

        fes_components: List[FESpace] = []

        # Iterate over each component and add a fes for each
        for component in self.model_components:
            if component not in ['u', 'p']:
                fes_components.append(getattr(ngs, self.element[component])(self.mesh, order=self.interp_ord,
                                                                            dirichlet=self.dirichlet_names.get(component, ''),
                                                                            dgjumps=self.DG))

        return ngs.FESpace([fes_u, fes_p] + fes_components, dgjumps=self.DG)

    def _construct_linearization_terms(self) -> Optional[List[GridFunction]]:
        tmp = GridFunction(self.fes.components[0])
        tmp.vec.data = self.IC.components[0].vec

        return [tmp]

    def update_linearization_terms(self, gfu: GridFunction) -> None:
        if self.linearize == 'Oseen':
            # Update the velocity linearization term.
            comp_index = self.model_components['u']
            self.W[comp_index].vec.data = gfu.components[comp_index].vec
        else:
            # Do nothing, no linearization term to update.
            pass

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        comp_index = self.model_components['u']

        if self.linearize == 'Oseen':
            if time_step > 0:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                w = U[comp_index]
            else:
                w = self.W[comp_index]
        elif self.linearize == 'IMEX':
            w = None
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_ODE_DIM(U, V, w, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_ODE_no_DIM(U, V, w, dt, time_step)

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        comp_index = self.model_components['u']

        if self.linearize == 'Oseen':
            if time_step > 0:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                w = U[comp_index]
            else:
                w = self.W[comp_index]
        elif self.linearize == 'IMEX':
            w = None
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_coefficient_DIM(U, V, w, dt)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_coefficient_no_DIM(U, V, w, dt, time_step)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                         dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:

        comp_index = self.model_components['u']

        if self.linearize == 'Oseen':
            if time_step > 0 and gfu_0 is not None:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                # Check for gfu_0 = None because adaptive_three_step requires a solve where w should be taken from W for
                # a half step (time_step != 0).
                w = gfu_0[comp_index]
            else:
                w = self.W[comp_index]
        elif self.linearize == 'IMEX':
            w = None
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

        if self.DIM:
            # Use the diffuse interface method.
            L = self._linear_DIM(V, w, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            L = self._linear_no_DIM(V, w, dt, time_step)

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                                dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:
        # Note that the time_step argument is not used because the nonlinear term in INS (convection) does not include
        # any boundary conditions or model parameters.

        if self.DIM:
            # Use the diffuse interface method.
            L = self._imex_explicit_DIM(V, gfu_0, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            L = self._imex_explicit_no_DIM(V, gfu_0, dt, time_step)

        return [L]

    def single_iteration(self, a: BilinearForm, L: LinearForm, precond: Preconditioner, gfu: GridFunction,
                         time_step: int = 0) -> None:

        if self.linearize == 'Oseen':
            # The component index representing velocity
            component = self.fes.components[self.model_components['u']]
            comp_index = self.model_components['u']

            # Number of linear iterations for this timestep
            num_iteration = 1

            # Boolean used to keep the while loop going
            done_iterating = False

            while not done_iterating:
                self.apply_dirichlet_bcs_to(gfu, time_step=time_step)

                a.Assemble()
                L.Assemble()
                precond.Update()

                self.construct_and_run_solver(a, L, precond, gfu)

                err = norm('l2_norm', self.W[comp_index], gfu.components[comp_index], self.mesh, component, average=False)
                gfu_norm = mean(gfu.components[comp_index], self.mesh)

                num_iteration += 1

                if self.verbose > 0:
                    print(num_iteration, err)

                self.W[comp_index].vec.data = gfu.components[comp_index].vec
                done_iterating = (err < self.abs_nonlinear_tolerance + self.rel_nonlinear_tolerance * gfu_norm) \
                                 or (num_iteration > self.nonlinear_max_iters)
        elif self.linearize == 'IMEX':
            self.construct_and_run_solver(a, L, precond, gfu)
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################
    def _bilinear_time_ODE_DIM(self, L: List[Union[ProxyFunction, GridFunction]], V: List[ProxyFunction],
                               w: GridFunction, dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables with time derivatives.
        """
        raise NotImplementedError

    def _bilinear_time_coefficient_DIM(self, U: List[ProxyFunction], V: List[ProxyFunction], w: GridFunction,
                                       time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables without time derivatives.
        """
        raise NotImplementedError

    def _bilinear_time_ODE_no_DIM(self, U: List[Union[ProxyFunction, GridFunction]], V: List[ProxyFunction],
                                  w: GridFunction, dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is not being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables with time derivatives.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure.
        u, p = U[0], U[1]
        v, q = V[0], V[1]

        # Flow profile domain integrals.
        a = dt * (
            self.kv[time_step] * InnerProduct(Grad(u), Grad(v))  # Stress, Newtonian
            ) * dx

        if self.linearize == 'Oseen':
            # Linearized convection term.
            a += -dt * InnerProduct(OuterProduct(u, w), Grad(v)) * dx

        # Transport of the mixture components.
        for i in self.extra_components_inverted:
            # Fix complaint about Optional int.
            i = cast(int, i)

            # The name of this component
            comp = self.extra_components_inverted[i]

            # Trial and test functions
            c = U[i]
            r = V[i]

            # Domain integrals.
            a += dt * self.Ds[comp][time_step] * InnerProduct(Grad(c), Grad(r)) * dx

            # TODO: Does not work
            # Surface reaction
            for marker in self.BC.get('surface_rxn', {}).get(comp, {}):
                # NOTE: bc_re_parse_dict is a dictionary which contains info about which bc terms contain
                #       a trial function. Thus if it contains a value, then that particular rxn term belongs
                #       in the bilinear form.
                if self.bc_functions.bc_re_parse_dict.get('surface_rxn', {}).get(comp, None) is not None:
                    a += -dt * self.BC['surface_rxn'][comp][marker][time_step] * r * self._ds(marker)

            # Bulk reaction
            # NOTE: model_functions_re_parse_dict is a dictionary which contains info about which source terms contain
            #       a trial function.
            if self.model_functions.model_functions_re_parse_dict.get('source', {}).get(comp, None) is not None:
                # NOTE: The negative sign is here since the source term was brought over to bilinear side,
                #       it usually does not have one since it is on the linear term side.
                # TODO: Temp solution to test, implement better once parser has been updated
                if self.BC.get('surface_rxn', {}).get(comp, None) is not None:
                    marker = list(self.BC['surface_rxn'][comp].keys())[0]
                    a += -dt * self.f[comp][time_step] * r * self._ds(marker)
                else:
                    a += -dt * self.f[comp][time_step] * r * dx

            if self.linearize == 'Oseen':
                # Neumann BC for C
                for marker in self.BC.get('neumann', {}).get(comp, {}):
                    h = self.BC['neumann'][comp][marker][time_step]
                    a += -dt * r.Trace() * (h - c * InnerProduct(w, n)) * self._ds(marker)

        # Flow profile boundary integrals.
        if self.DG:
            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += dt * (
                    self.kv[time_step] * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - self.kv[time_step] * InnerProduct(Grad(u), OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(u, n))  # 1/2 of penalty for u=g on
                    ) * self._ds(self.dirichlet_names['u'])

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection term)
                    a += dt * v * (0.5 * w * n * u + 0.5 * Norm(w * n) * u) * self._ds(self.dirichlet_names['u'])

        if self.linearize == 'Oseen':
            # Stress needs a no-backflow component in the bilinear form.
            for marker in self.BC.get('stress', {}).get('stress', {}):
                if self.DG:
                    a += dt * v * (IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
                else:
                    a += dt * v.Trace() * (IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Parallel Flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += dt * v * (u - n * InnerProduct(u, n)) * self._ds(marker)
            else:
                a += dt * v.Trace() * (u.Trace() - n * InnerProduct(u.Trace(), n)) * self._ds(marker)

        return a

    def _bilinear_time_coefficient_no_DIM(self, U: List[ProxyFunction], V: List[ProxyFunction], w: GridFunction,
                                          dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is not being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables without time derivatives.
        """
        # TODO: Not sure that this has actually been split up correctly. Convection term is currently the only term that must always be solved implicitly.
        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure.
        u, p = U[0], U[1]
        v, q = V[0], V[1]

        # Flow profile domain integrals.
        a = dt * (
            - div(u) * q  # Conservation of mass
            - div(v) * p  # Pressure
            - 1e-10 * p * q  # Stabilization term
            ) * dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a += dt * (
                self.kv[time_step] * alpha * InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I from ∇u^
                - self.kv[time_step] * InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                - self.kv[time_step] * InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                ) * dx(skeleton=True)

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                a += dt * jump_v * (w * n * avg_u + 0.5 * Norm(w * n) * jump_u) * dx(skeleton=True)

        # Transport of mixture components.
        for i in self.extra_components_inverted:
            # Fix complaint about Optional int.
            i = cast(int, i)

            # Trial and test functions
            c = U[i]
            r = V[i]

            if self.linearize == 'Oseen':
                # Additional convection term.
                a += -dt * c * InnerProduct(w, Grad(r)) * dx

        return a

    def _linear_DIM(self, V: List[ProxyFunction], w: GridFunction, dt: Parameter, time_step: int) -> LinearForm:
        """
        Linear form when the diffuse interface method is being used. Handles both CG and DG.
        """
        raise NotImplementedError

    def _imex_explicit_DIM(self, V: List[ProxyFunction], gfu_u: List[GridFunction], dt: Parameter, time_step: int) \
            -> LinearForm:
        """
        Constructs the explicit IMEX terms for the linear form when the diffuse interface method is being used.
        Handles both CG and DG.
        """
        raise NotImplementedError

    def _linear_no_DIM(self, V: List[ProxyFunction], w: GridFunction, dt: Parameter, time_step: int) -> LinearForm:
        """
        Linear form when the diffuse interface method is not being used. Handles both CG and DG.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the test functions for velocity and pressure.
        v, q = V[0], V[1]

        # Flow profile domain integrals.
        L = dt * v * self.f['u'][time_step] * dx

        # Transport of mixture components.
        for i in self.extra_components_inverted:
            # Fix complaint about Optional int.
            i = cast(int, i)

            # The name of this component
            comp = self.extra_components_inverted[i]

            # Test function
            r = V[i]

            # Total Flux BC
            for marker in self.BC.get('total_flux', {}).get(comp, {}):
                h = self.BC['total_flux'][comp][marker][time_step]
                L += dt * r.Trace() * h * self._ds(marker)

            # Surface reaction
            for marker in self.BC.get('surface_rxn', {}).get(comp, {}):
                # NOTE: bc_re_parse_dict is a dictionary which contains info about which bc terms contain
                #       a trial function. Thus if it contains a value, then that particular rxn term belongs
                #       in the bilinear form.
                if self.bc_functions.bc_re_parse_dict.get('surface_rxn', {}).get(comp, None) is None:
                    L += -dt * self.BC['surface_rxn'][comp][marker][time_step] * r * self._ds(marker)

            # Bulk reaction
            # NOTE: model_functions_re_parse_dict is a dictionary which contains info about which source terms contain
            #       a trial function.
            if self.model_functions.model_functions_re_parse_dict.get('source', {}).get(comp, None) is None:
                # TODO: Temp solution to test, implement better once parser has been updated
                if self.BC.get('surface_rxn', {}).get(comp, None) is not None:
                    marker = list(self.BC['surface_rxn'][comp].keys())[0]
                    L += dt * self.f[comp][time_step] * r * self._ds(marker)
                else:
                    L += dt * self.f[comp][time_step] * r * dx

        # Dirichlet BC for u
        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * (
                    self.kv[time_step] * alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(g, n))  # 1/2 of penalty for u=g
                ) * self._ds(marker)

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection)
                    L += dt * v * (-0.5 * w * n * g + 0.5 * Norm(w * n) * g) * self._ds(marker)

        # Stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        return L

    def _imex_explicit_no_DIM(self, V: List[ProxyFunction], gfu_0: List[GridFunction], dt: Parameter, time_step: int) \
            -> LinearForm:
        """
        Constructs the explicit IMEX terms for the linear form when the diffuse interface method is not being used.
        Handles both CG and DG.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the test functions for velocity and pressure.
        v, q = V[0], V[1]

        # Velocity term
        gfu_u = gfu_0[0]

        # Flow profile domain integrals.
        L = -dt * InnerProduct(Grad(gfu_u) * gfu_u, v) * ngs.dx

        for i in self.extra_components_inverted:
            # Fix complaint about Optional int.
            i = cast(int, i)

            # The name of this component
            comp = self.extra_components_inverted[i]

            # Previous concentration result
            c_prev = gfu_0[i]

            # Test function
            r = V[i]

            # Convection term
            L += -dt * c_prev * InnerProduct(gfu_u, Grad(r)) * dx

            # Neumann BC for C
            # TODO: Is this implemented correctly?
            for marker in self.BC.get('neumann', {}).get(comp, {}):
                h = self.BC['neumann'][comp][marker][time_step]
                L += -dt * r.Trace() * h * self._ds(marker)

            # Check that a total flux BC has not been incorrectly specified
            if self.BC.get('total_flux', {}).get(comp, None) is not None:
                raise ValueError('Cannot specify total flux for IMEX schemes since convection term is fully explicit.')

        return L

