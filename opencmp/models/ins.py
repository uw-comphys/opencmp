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

import logging
from typing import Dict, List, Tuple, Optional, Union

import ngsolve
from ngsolve.comp import ProxyFunction
from ngsolve import Grad, HDiv, IfPos, InnerProduct, Norm, OuterProduct, Parameter, GridFunction, FESpace, BilinearForm, \
    LinearForm, \
    Preconditioner, div, dx

from ..helpers.ngsolve_ import get_special_functions
from ..helpers.dg import avg, jump, grad_avg
from . import Model
from ..helpers.error import norm, mean


class INS(Model):
    """
    A single phase incompressible Navier-Stokes model.
    """

    def _pre_init(self) -> None:
        # Nothing needs to be done
        pass

    def _post_init(self) -> None:
        # Load the solver parameters.

        # indicate that the model is nonlinear
        self.nonlinear = True

        self.linearize = self.config.get_item(['SOLVER', 'linearization_method'], str)
        if self.linearize not in ['Oseen', 'IMEX']:
            self.linearize = 'Oseen'
            logging.info('Linearization method not specified, using Oseen by default.')

        if self.linearize == 'Oseen':
            nonlinear_tolerance = self.config.get_dict(['SOLVER', 'nonlinear_tolerance'], self.run_dir, None)
            self.abs_nonlinear_tolerance = nonlinear_tolerance['absolute']
            self.rel_nonlinear_tolerance = nonlinear_tolerance['relative']
            self.nonlinear_max_iters = self.config.get_item(['SOLVER', 'nonlinear_max_iterations'], int)
            if self.nonlinear_max_iters < 1:
                raise ValueError('Nonlinear solve must involve at least one iteration. Set nonlinear_max_iterations to '
                                 'at least 1.')
            self.W = self._construct_linearization_terms()

    def _define_model_components(self) -> Dict[str, Optional[int]]:
        return {'u': 0,
                'p': 1}

    def _define_model_local_error_components(self) -> Dict[str, bool]:
        return {'u': True,
                'p': False}

    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        return [
            {'u': True,
             'p': False}
        ]

    def _define_num_weak_forms(self) -> int:
        return 1

    def _define_bc_types(self) -> List[str]:
        return ['dirichlet', 'stress', 'parallel', 'pinned']

    @staticmethod
    def allows_explicit_schemes() -> bool:
        # INS cannot work with explicit schemes
        return False

    def _set_model_parameters(self) -> None:
        self.kv: Dict = self.model_functions.model_parameters_dict['kinematic_viscosity']['all']
        self.f: Dict  = self.model_functions.model_functions_dict['source']
        if self.fixed_velocity and 'u' in self.f:
            # Remove the source term for the conservation of momentum if it's not being solved.
            self.f.pop('u')

    def _construct_fes(self) -> FESpace:
        return FESpace(self._construct_fes_helper(), dgjumps=self.DG)

    def _construct_fes_helper(self) -> List[FESpace]:
        """
        Helper function for creating the FESpace.

        This function exists in order to allow multicomponent_ins (and any further additions) to simplify their FES
        creation by being able to use this helper function to create the finite element space for velocity and pressure.

        Return:
            A list containing individual finite element spaces
        """
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
            fes_u = getattr(ngsolve, self.element['u'])(self.mesh, order=self.interp_ord,
                                                        dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)

        if self.element['p'] == 'L2' and 'p' in self.dirichlet_names:
            raise ValueError('Not able to pin pressure at a point on L2 spaces.')
        else:
            fes_p = getattr(ngsolve, self.element['p'])(self.mesh, order=self.interp_ord - 1,
                                                        dirichlet=self.dirichlet_names.get('p', ''), dgjumps=self.DG)

        return [fes_u, fes_p]

    def _construct_linearization_terms(self) -> Optional[List[GridFunction]]:
        tmp = GridFunction(self._construct_ic_fes().components[0])  # Read a new FES
        tmp.vec.data = self.IC.components[0].vec

        return [tmp]

    def _get_wind(self, gfu: Union[List[ProxyFunction], List[GridFunction]], time_step: int) -> Optional[ProxyFunction]:
        """
        Function to obtain the wind term used to linearize the convection term.

        Args:
            gfu:        List of
            time_step:  What time step values to use for ex: boundary conditions.
                        The value corresponds to the index of the time step in t_param and dt_param.

        Returns:
            The wind/linearization term used for linearizing the convection term.
        """

        if self.linearize == 'Oseen':
            # gfu = None: intermediary step from adaptive_three_step requires w from W when time_step > 0
            if time_step > 0 and gfu is not None:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                w = gfu[self.model_components['u']]
            else:
                w = self.W[self.model_components['u']]
        elif self.linearize == 'IMEX':
            w = None
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

        return w

    def update_linearization(self, gfu: GridFunction) -> None:
        if self.linearize == 'Oseen':
            # Update the velocity linearization term.
            try:
                # First try just updating the value of the term.
                self.W[self.model_components['u']].vec.data = gfu.components[self.model_components['u']].vec
            except:
                # In some cases the finite element space of the model will have changed, so the linearization term needs
                # to be completely reconstructed.
                # E.g. during convergence testing.
                self.W = self._construct_linearization_terms()
        else:
            # Do nothing, no linearization term to update.
            pass

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        w = self._get_wind(U, time_step)

        # Define the special DG functions
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test function for velocity
        u = U[self.model_components['u']]
        v = V[self.model_components['u']]

        # Domain integrals. Newtonian Stress
        a = dt * (self.kv[time_step] * InnerProduct(Grad(u), Grad(v))) * dx

        if self.linearize == 'Oseen':
            # Linearized convection term.
            a += -dt * InnerProduct(OuterProduct(u, w), Grad(v)) * dx

        if self.DG:
            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += dt * (
                        self.kv[time_step] * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                        - self.kv[time_step] * InnerProduct(Grad(u), OuterProduct(v, n))  # ∇u^ = ∇u
                        - self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                ) * self._ds(self.dirichlet_names['u'])

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection term)
                    a += dt * v * (0.5 * w * n * u + 0.5 * Norm(w * n) * u) * self._ds(self.dirichlet_names['u'])

        if self.linearize == 'Oseen':
            # Stress needs a no-backflow component in the bilinear form.
            for marker in self.BC.get('stress', {}).get('u', {}):
                if self.DG:
                    a += dt * v * (IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
                else:
                    a += dt * v.Trace() * (IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Parallel Flow BC
        for marker in self.BC.get('parallel', {}).get('u', {}):
            if self.DG:
                a += dt * v * (u - n * InnerProduct(u, n)) * self._ds(marker)
            else:
                a += dt * v.Trace() * (u.Trace() - n * InnerProduct(u.Trace(), n)) * self._ds(marker)

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:

        w = self._get_wind(U, time_step)

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure.
        u, p = U[self.model_components['u']], U[self.model_components['p']]
        v, q = V[self.model_components['u']], V[self.model_components['p']]

        # Domain integrals.
        a = dt * (               # TODO: Why is conservation of mass multiplied by a time-step??
                - div(u) * q     # Conservation of mass  TODO: Why this here?
                - div(v) * p     # Pressure
                - 1e-10 * p * q  # Stabilization term   TODO: This should not be needed (at least not for DG)
        ) * dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            # TODO: Why are these in here?
            #       James believes that these operations (jump, grad, etc.) may be returning a known value, one in terms
            #       of previous values (timestep/iteration/something) of the operand.
            #       E.g. p is an unknown, grad(p) would be fixed in terms of a previous value of p and could NOT be
            #       solved for.
            a += dt * (
                    self.kv[time_step] * alpha * InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I from ∇u^
                    - self.kv[time_step] * InnerProduct(avg_grad_u, OuterProduct(jump_v, n))  # Stress
                    - self.kv[time_step] * InnerProduct(avg_grad_v, OuterProduct(jump_u, n))  # U
            ) * dx(skeleton=True)

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                a += dt * jump_v * (w * n * avg_u + 0.5 * Norm(w * n) * jump_u) * dx(skeleton=True)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:

        w = self._get_wind(gfu_0, time_step)

        # Define the special DG functions.
        n, h, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Domain integrals.
        L = dt * v * self.f['u'][time_step] * dx

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
        for marker in self.BC.get('stress', {}).get('u', {}):
            h = self.BC['stress']['u'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Velocity linearization term
        gfu_u = gfu_0[self.model_components['u']]

        # Linearized convection term.
        L = -dt * InnerProduct(Grad(gfu_u) * gfu_u, v) * dx

        return [L]

    def solve_single_step(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                         precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:
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

                a_lst[0].Assemble()
                L_lst[0].Assemble()
                if precond_lst[0] is not None:
                    precond_lst[0].Update()

                self.linear_solve(a_lst[0], L_lst[0], precond_lst[0], gfu)

                err = norm("l2_norm", self.W[comp_index], gfu.components[comp_index], self.mesh, component, average=False)
                gfu_norm = mean(gfu.components[comp_index], self.mesh)

                num_iteration += 1

                if self.verbose > 0:
                    print(num_iteration, err)

                self.W[comp_index].vec.data = gfu.components[comp_index].vec
                done_iterating = (err < self.abs_nonlinear_tolerance + self.rel_nonlinear_tolerance * gfu_norm) or (num_iteration > self.nonlinear_max_iters)
        elif self.linearize == 'IMEX':
            self.linear_solve(a_lst[0], L_lst[0], precond_lst[0], gfu)
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

    def linearized_solve(self, a_assembled: BilinearForm, L_assembled: LinearForm, precond: Preconditioner, gfu: GridFunction) -> Tuple[float, float]:
        if self.linearize == 'Oseen':
            # The component index representing velocity
            component = self.fes.components[self.model_components['u']]
            comp_index = self.model_components['u']

            # not sure how to handle the dependence of BC on time currently
            self.apply_dirichlet_bcs_to(gfu, time_step=0)

            # solver linearized model
            self.linear_solve(a_assembled, L_assembled, precond, gfu)

            # calculate error and norm of solution
            err = norm("l2_norm", self.W[comp_index], gfu.components[comp_index], self.mesh, component, average=False)
            gfu_norm = mean(gfu.components[comp_index], self.mesh)

            self.W[comp_index].vec.data = gfu.components[comp_index].vec

            return(err, gfu_norm)

        elif self.linearize == 'IMEX':
            logging.error('Linearization scheme \"{}\" is appropriate for stationary solve.'.format(self.linearize))
            raise ValueError('Linearization scheme \"{}\" is appropriate for stationary solve.'.format(self.linearize))
        else:
            logging.error('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))
