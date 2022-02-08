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
from ..helpers.ngsolve_ import get_special_functions
from ..helpers.dg import avg, jump, grad_avg
from . import Model
from typing import Dict, List, Optional, Union
from ngsolve.comp import ProxyFunction
from ngsolve import Parameter, GridFunction, FESpace, BilinearForm, LinearForm, Preconditioner
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
        self.linearize = self.config.get_item(['SOLVER', 'linearization_method'], str)
        if self.linearize not in ['Oseen', 'IMEX']:
            raise TypeError('Don\'t recognize the linearization method.')

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
                'p': True}

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
        self.kv = self.model_functions.model_parameters_dict['kinematic_viscosity']['all']
        self.f = self.model_functions.model_functions_dict['source']

    def _construct_fes(self) -> FESpace:
        return ngs.FESpace(self._contruct_fes_helper(), dgjumps=self.DG)

    def _contruct_fes_helper(self) -> List[FESpace]:
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
            fes_u = ngs.HDiv(self.mesh, order=self.interp_ord, dirichlet=self.dirichlet_names.get('u', ''),
                             dgjumps=self.DG, RT=True)
        else:
            fes_u = getattr(ngs, self.element['u'])(self.mesh, order=self.interp_ord,
                                                    dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)

        if self.element['p'] == 'L2' and 'p' in self.dirichlet_names:
            raise ValueError('Not able to pin pressure at a point on L2 spaces.')
        else:
            fes_p = getattr(ngs, self.element['p'])(self.mesh, order=self.interp_ord - 1,
                                                    dirichlet=self.dirichlet_names.get('p', ''), dgjumps=self.DG)

        return [fes_u, fes_p]

    def _construct_linearization_terms(self) -> Optional[List[GridFunction]]:
        tmp = ngs.GridFunction(self.fes.components[0])
        tmp.vec.data = self.IC.components[0].vec

        return [tmp]

    def update_linearization_terms(self, gfu: GridFunction) -> None:
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

        if self.linearize == 'Oseen':
            if time_step > 0:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                w = U[self.model_components['u']]
            else:
                w = self.W[self.model_components['u']]
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

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:

        if self.linearize == 'Oseen':
            if time_step > 0:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                w = U[self.model_components['u']]
            else:
                w = self.W[self.model_components['u']]
        elif self.linearize == 'IMEX':
            w = None
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_coefficient_DIM(U, V, w, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_coefficient_no_DIM(U, V, w, dt, time_step)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:

        if self.linearize == 'Oseen':
            if time_step > 0 and gfu_0 is not None:
                # Using known values from a previous time step, so no need to iteratively solve for a wind.
                # Check for gfu_0 = None because adaptive_three_step requires a solve where w should be taken from W for
                # a half step (time_step != 0).
                w = gfu_0[self.model_components['u']]
            else:
                w = self.W[self.model_components['u']]
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

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:

        if self.DIM:
            # Use the diffuse interface method.
            L = self._imex_explicit_DIM(V, gfu_0, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            L = self._imex_explicit_no_DIM(V, gfu_0, dt, time_step)

        return [L]

    def solve_single_step(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                         precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:
        if self.linearize == 'Oseen':
            # The component index representing velocity
            component = self.fes.components[self.model_components['u']]
            comp_index = self.model_components['u']

            # Number of linear iterations for this timestep
            num_iteration = 0

            # Boolean used to keep the while loop going
            done_iterating = False

            while not done_iterating:
                self.apply_dirichlet_bcs_to(gfu, time_step=time_step)

                a_lst[0].Assemble()
                L_lst[0].Assemble()
                if precond_lst[0] is not None:
                    precond_lst[0].Update()

                self.construct_and_run_solver(a_lst[0], L_lst[0], precond_lst[0], gfu)

                err = norm("l2_norm", self.W[comp_index], gfu.components[comp_index], self.mesh, component, average=False)
                gfu_norm = mean(gfu.components[comp_index], self.mesh)

                num_iteration += 1

                if self.verbose:
                    print('Nonlinear iteration {}, L2 error in velocity = {}'.format(num_iteration, err))

                self.W[comp_index].vec.data = gfu.components[comp_index].vec
                done_iterating = (err < self.abs_nonlinear_tolerance + self.rel_nonlinear_tolerance * gfu_norm) or (num_iteration > self.nonlinear_max_iters)
        elif self.linearize == 'IMEX':
            self.construct_and_run_solver(a_lst[0], L_lst[0], precond_lst[0], gfu)
        else:
            raise ValueError('Linearization scheme \"{}\" is not implemented.'.format(self.linearize))

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################

    def _bilinear_time_ODE_DIM(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                               w: Optional[GridFunction], dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.

        This is the portion of the bilinear form which contains variables WITH time derivatives.

        Args:
            U: A list of trial functions for the model's finite element space, or a list of grid functions containing
                the previous time step's solution.
            V: A list of test (weighting) functions for the model's finite element space.
            w: Velocity linearization term if using Oseen linearization, None otherwise
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test function for velocity
        u = U[self.model_components['u']]
        v = V[self.model_components['u']]

        # Domain integrals.
        a = dt * (
            self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            ) * self.DIM_solver.phi_gfu * ngs.dx

        if self.linearize == 'Oseen':
            # Linearized convection term.
            a += -dt * ngs.InnerProduct(ngs.OuterProduct(u, w), ngs.Grad(v)) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u to zero where phi is zero. If DIM rigid body motion is being used force u to the velocity of the
        # rigid body instead.
        a += dt * alpha * u * v * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        if self.DG:
            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += dt * (
                    self.kv[time_step] * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv[time_step] * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                    ) * self._ds(self.dirichlet_names['u'])

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection term)
                    a += dt * v * (0.5 * w * n * u + 0.5 * ngs.Norm(w * n) * u) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += dt * (
                self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, self.DIM_solver.grad_phi_gfu))
                + self.kv[time_step] * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, self.DIM_solver.grad_phi_gfu))
                + self.kv[time_step] * alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

            if self.linearize == 'Oseen':
                # Additional convection term penalty.
                a += dt * (v * (0.5 * w * -self.DIM_solver.grad_phi_gfu * u + 0.5
                                * ngs.Norm(w * -self.DIM_solver.grad_phi_gfu) * u)
                    ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        if self.linearize == 'Oseen':
            # Conformal stress BC needs a no-backflow component in the bilinear form.
            for marker in self.BC.get('stress', {}).get('stress', {}):
                if self.DG:
                    a += dt * v * (ngs.IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
                else:
                    a += dt * v.Trace() * (ngs.IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Conformal parallel flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += dt * v * (u - n * ngs.InnerProduct(u, n)) * self._ds(marker)
            else:
                a += dt * v.Trace() * (u.Trace() - n * ngs.InnerProduct(u.Trace(), n)) * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return a

    def _bilinear_time_coefficient_DIM(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                       w: Optional[GridFunction], dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.

        This is the portion of the bilinear form which contains variables WITHOUT time derivatives.

        Args:
            U: A list of trial functions for the model's finite element space.
            V: A list of test (weighting) functions for the model's finite element space.
            w: Velocity linearization term if using Oseen linearization, None otherwise
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure.
        u, p = U[self.model_components['u']], U[self.model_components['p']]
        v, q = V[self.model_components['u']], V[self.model_components['p']]

        # Domain integrals.
        a = dt * (
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            - 1e-10 * p * q   # Stabilization term
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force grad(p) to zero where phi is zero. If rigid body motion is being used ignore this.
        if not self.DIM_solver.rigid_body_motion:
            a += -dt * p * (ngs.div(v)) * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a += dt * (
                self.kv[time_step] * alpha * ngs.InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I from ∇u^
                - self.kv[time_step] * ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                - self.kv[time_step] * ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                a += dt * jump_v * (w * n * avg_u + 0.5 * ngs.Norm(w * n) * jump_u) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

        return a

    def _bilinear_time_ODE_no_DIM(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                  w: Optional[GridFunction], dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is NOT being used. Handles both CG and DG.

        This is the portion of the bilinear form which contains variables WITH time derivatives.

        Args:
            U: A list of trial functions for the model's finite element space, or a list of grid functions containing
                the previous time step's solution.
            V: A list of test (weighting) functions for the model's finite element space.
            w: Velocity linearization term if using Oseen linearization, None otherwise
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Define the special DG functions
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test function for velocity
        u = U[self.model_components['u']]
        v = V[self.model_components['u']]

        # Domain integrals.
        a = dt * (
            self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            ) * ngs.dx

        if self.linearize == 'Oseen':
            # Linearized convection term.
            a += -dt * ngs.InnerProduct(ngs.OuterProduct(u, w), ngs.Grad(v)) * ngs.dx

        if self.DG:
            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += dt * (
                    self.kv[time_step] * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv[time_step] * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on
                    ) * self._ds(self.dirichlet_names['u'])

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection term)
                    a += dt * v * (0.5 * w * n * u + 0.5 * ngs.Norm(w * n) * u) * self._ds(self.dirichlet_names['u'])

        if self.linearize == 'Oseen':
            # Stress needs a no-backflow component in the bilinear form.
            for marker in self.BC.get('stress', {}).get('stress', {}):
                if self.DG:
                    a += dt * v * (ngs.IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
                else:
                    a += dt * v.Trace() * (ngs.IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Parallel Flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += dt * v * (u - n * ngs.InnerProduct(u, n)) * self._ds(marker)
            else:
                a += dt * v.Trace() * (u.Trace() - n * ngs.InnerProduct(u.Trace(), n)) * self._ds(marker)

        return a

    def _bilinear_time_coefficient_no_DIM(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                          w: Optional[GridFunction], dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is NOT being used. Handles both CG and DG.

        This is the portion of the bilinear form which contains variables WITHOUT time derivatives.

        Args:
            U: A list of trial functions for the model's finite element space.
            V: A list of test (weighting) functions for the model's finite element space.
            w: Velocity linearization term if using Oseen linearization, None otherwise.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure.
        u, p = U[self.model_components['u']], U[self.model_components['p']]
        v, q = V[self.model_components['u']], V[self.model_components['p']]

        # Domain integrals.
        a = dt * (
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            - 1e-10 * p * q   # Stabilization term
            ) * ngs.dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a += dt * (
                self.kv[time_step] * alpha * ngs.InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I from ∇u^
                - self.kv[time_step] * ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                - self.kv[time_step] * ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                ) * ngs.dx(skeleton=True)

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                a += dt * jump_v * (w * n * avg_u + 0.5 * ngs.Norm(w * n) * jump_u) * ngs.dx(skeleton=True)

        return a

    def _linear_DIM(self, V: List[ProxyFunction], w: Optional[GridFunction], dt: Parameter, time_step: int)\
            -> LinearForm:
        """
        Linear form when the diffuse interface method is being used. Handles both CG and DG.

        Args:
            V: A list of test (weighting) functions for the model's finite element space.
            w: Velocity linearization term if using Oseen linearization, None otherwise.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The linear form.
        """

        # Define the special DG functions.
        n, h, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Domain integrals.
        L = dt * v * self.f['u'][time_step] * self.DIM_solver.phi_gfu * ngs.dx

        # Force u to zero where phi is zero. If DIM rigid body motion is being used force u to the velocity of the
        # rigid body instead.
        if self.DIM_solver.rigid_body_motion:
            for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
                g = self.DIM_BC['dirichlet']['u'][marker][time_step]
                L += dt * alpha * g * v * (1.0 - self.DIM_solver.phi_gfu) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        if self.DG:
            # Conformal Dirichlet BCs for u.
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * (
                    self.kv[time_step] * alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - self.kv[time_step] * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    ) * self._ds(marker)

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection)
                    L += dt * v * (-0.5 * w * n * g + 0.5 * ngs.Norm(w * n) * g) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            g = self.DIM_BC['dirichlet']['u'][marker][time_step]
            L += dt * (
                self.kv[time_step] * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, self.DIM_solver.grad_phi_gfu))
                + self.kv[time_step] * alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
            ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                L += dt * v * (0.5 * w * self.DIM_solver.grad_phi_gfu * g + 0.5
                               * ngs.Norm(w * -self.DIM_solver.grad_phi_gfu) * g) \
                     * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return L

    def _linear_no_DIM(self, V: List[ProxyFunction], w: Optional[GridFunction], dt: Parameter, time_step: int)\
            -> LinearForm:
        """
        Linear form when the diffuse interface method is NOT being used. Handles both CG and DG.

        Args:
            V: A list of test (weighting) functions for the model's finite element space.
            w: Velocity linearization term if using Oseen linearization, None otherwise.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The linear form.
        """

        # Define the special DG functions.
        n, h, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Domain integrals.
        L = dt * v * self.f['u'][time_step] * ngs.dx

        # Dirichlet BC for u
        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * (
                    self.kv[time_step] * alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - self.kv[time_step] * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                ) * self._ds(marker)

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection)
                    L += dt * v * (-0.5 * w * n * g + 0.5 * ngs.Norm(w * n) * g) * self._ds(marker)

        # Stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        return L

    def _imex_explicit_DIM(self, V: List[ProxyFunction], gfu_0: List[GridFunction], dt: Parameter, time_step: int)\
            -> LinearForm:
        """
        Contains any linear form terms resulting from the linearization of terms due to the IMEX method.

        For when the diffuse interface method is being used. Handles both CG and DG.

        Args:
            V: A list of test (weighting) functions for the model's finite element space.
            gfu_0: The previous time step's solution.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the linear form.
        """

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Velocity linearization term
        gfu_u = gfu_0[self.model_components['u']]

        # Linearized convection term.
        L = -dt * ngs.InnerProduct(ngs.Grad(gfu_u) * gfu_u, v) * self.DIM_solver.phi_gfu * ngs.dx

        return L

    def _imex_explicit_no_DIM(self, V: List[ProxyFunction], gfu_0: List[GridFunction], dt: Parameter, time_step: int)\
            -> LinearForm:
        """
        Contains any linear form terms resulting from the linearization of terms due to the IMEX method.

        For when the diffuse interface method is NOT being used. Handles both CG and DG.

        Args:
            V: A list of test (weighting) functions for the model's finite element space.
            gfu_0: The previous time step's solution.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the linear form.
        """

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Velocity linearization term
        gfu_u = gfu_0[self.model_components['u']]

        # Linearized convection term.
        L = -dt * ngs.InnerProduct(ngs.Grad(gfu_u) * gfu_u, v) * ngs.dx

        return L
