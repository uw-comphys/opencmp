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
from ..helpers.dg import jump, grad_avg
from . import Model
from typing import Dict, List, Union, Optional
from ngsolve.comp import FESpace, ProxyFunction
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm, Preconditioner


class Poisson(Model):
    """
    A poisson model.
    """

    def _pre_init(self) -> None:
        # Nothing needs to be done
        pass

    def _post_init(self) -> None:
        # Nothing needs to be done
        pass

    def _define_model_components(self) -> Dict[str, Optional[int]]:
        return {'u': 0}

    def _define_model_local_error_components(self) -> Dict[str, bool]:
        return {'u': True}

    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        return [{'u': True}]

    def _define_num_weak_forms(self) -> int:
        return 1

    def _define_bc_types(self) -> List[str]:
        return ['robin', 'dirichlet', 'neumann']

    @staticmethod
    def allows_explicit_schemes() -> bool:
        return True

    def _construct_fes(self) -> FESpace:
        fes = getattr(ngs, self.element['u'])(self.mesh, order=self.interp_ord,
                                             dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)
        compound_fes = ngs.FESpace([fes])

        return compound_fes

    def _set_model_parameters(self) -> None:
        self.dc = self.model_functions.model_parameters_dict['diffusion_coefficient']['all']
        self.f = self.model_functions.model_functions_dict['source']['all']

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        # Split up the trial (weighting) and test functions
        u = U[0]
        v = V[0]

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_ODE_DIM(u, v, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_ODE_no_DIM(u, v, dt, time_step)

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:
        # Split up the trial (weighting) and test functions
        u = U[0]
        v = V[0]

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_coefficient_DIM(u, v, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_coefficient_no_DIM(u, v, dt, time_step)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:
        # V is a list of length 1 for Poisson, get just the first element
        v = V[0]

        if self.DIM:
            # Use the diffuse interface method.
            L = self._linear_DIM(v, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            L = self._linear_no_DIM(v, dt, time_step)

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:
        # The Poisson equation is linear and has no need for IMEX schemes.
        pass

    def solve_single_step(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                         precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:

        self.construct_and_run_solver(a_lst[0], L_lst[0], precond_lst[0], gfu)

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################

    def _bilinear_time_ODE_DIM(self, u: Union[ProxyFunction, GridFunction], v: ProxyFunction, dt: Parameter,
                               time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.

        This is the portion of the bilinear form for model variables with time derivatives.

        Args:
            u: The trial function for the model's finite element space, or a grid function containing the previous time
                step's solution.
            v: The test (weighting) function for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Laplacian term
        a = dt * self.dc[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u to zero where phi is zero.
        a += dt * u * v * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += dt * self.dc[time_step] * (
                    - u * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    - ngs.Grad(u) * n * v  # ∇u^ = ∇u
                    + alpha * u * v        # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += dt * self.dc[time_step] * (
                ngs.Grad(u) * self.DIM_solver.grad_phi_gfu * v
                + ngs.Grad(v) * self.DIM_solver.grad_phi_gfu * u
                + alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # DIM Robin BCs for u.
        for marker in self.DIM_BC.get('robin', {}).get('u', {}):
            r, q = self.DIM_BC['robin']['u'][marker][time_step]
            a += dt * self.dc[time_step] * (
                r * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker][time_step]
            a += -dt * self.dc[time_step] * r * u * v * self._ds(marker)

        return a

    def _bilinear_time_coefficient_DIM(self, u: ProxyFunction, v: ProxyFunction, dt: Parameter, time_step: int) \
            -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.

        This is the portion of the bilinear form for model variables without time derivatives.

        Args:
            u: The trial function for the model's finite element space.
            v: The test (weighting) function for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        if self.DG:
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a = dt * self.dc[time_step] * (
                - jump_u * n * avg_grad_v  # U
                - avg_grad_u * jump_v * n  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                + alpha * jump_u * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

        else:
            a = ngs.CoefficientFunction(0.0) * ngs.dx

        return a

    def _bilinear_time_ODE_no_DIM(self, u: Union[ProxyFunction, GridFunction], v: ProxyFunction, dt: Parameter,
                                  time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is not being used. Handles both CG and DG.

        This is the portion of the bilinear form for model variables with time derivatives.

        Args:
            u: The trial function for the model's finite element space, or a grid function containing the previous time
                step's solution.
            v: The test (weighting) function for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        # Laplacian term
        a = dt * self.dc[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            # Define the special DG functions.
            n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for Dirichlet BCs
                a += dt * self.dc[time_step] * (
                    - u * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    - ngs.Grad(u) * n * v  # ∇u^ = ∇u
                    + alpha * u * v        # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker][time_step]
            a += -dt * self.dc[time_step] * r * u * v * self._ds(marker)

        return a

    def _bilinear_time_coefficient_no_DIM(self, u: ProxyFunction, v: ProxyFunction, dt: Parameter, time_step: int) \
            -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is not being used. Handles both CG and DG.

        This is the portion of the bilinear form for model variables without time derivatives.

        Args:
            u: The trial function for the model's finite element space.
            v: The test (weighting) function for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The described portion of the bilinear form.
        """

        if self.DG:
            # Define the special DG functions.
            n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a = dt * self.dc[time_step] * (
                - jump_u * n * avg_grad_v  # U
                - avg_grad_u * n * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                + alpha * jump_u * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                ) * ngs.dx(skeleton=True)

        else:
            a = ngs.CoefficientFunction(0.0) * ngs.dx

        return a

    def _linear_DIM(self, v: ProxyFunction, dt: Parameter, time_step: int) -> LinearForm:
        """
        Linear form when the diffuse interface method is being used. Handles both CG and DG.

        Args:
            v: The test (weighting) function for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The linear form.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Source term
        L = dt * self.f[time_step] * v * self.DIM_solver.phi_gfu * ngs.dx

        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                # Penalty terms for conformal Dirichlet BCs
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * self.dc[time_step] * (
                    alpha * g * v          # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - g * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    ) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            # Penalty terms for DIM Dirichlet BCs
            g = self.DIM_BC['dirichlet']['u'][marker][time_step]
            L += dt * self.dc[time_step] * (
                ngs.Grad(v) * self.DIM_solver.grad_phi_gfu * g      # TODO
                + alpha * g * v * self.DIM_solver.mag_grad_phi_gfu  # TODO
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # DIM Neumann BCs for u.
        for marker in self.DIM_BC.get('neumann', {}).get('u', {}):
            h = self.DIM_BC['neumann']['u'][marker][time_step]
            L += dt * self.dc[time_step] * (-h * v * self.DIM_solver.mag_grad_phi_gfu
                            ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # DIM Robin BCs for u.
        for marker in self.DIM_BC.get('robin', {}).get('u', {}):
            r, q = self.DIM_BC['robin']['u'][marker][time_step]
            L += dt * self.dc[time_step] * (r * q * v * self.DIM_solver.mag_grad_phi_gfu
                            ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Neumann BCs for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker][time_step]
            L += dt * self.dc[time_step] * h * v * self._ds(marker)

        # Conformal Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker][time_step]
            L += -dt * self.dc[time_step] * r * q * v * self._ds(marker)

        return L

    def _linear_no_DIM(self, v: ProxyFunction, dt: Parameter, time_step: int) -> LinearForm:
        """
        Linear form when the diffuse interface method is not being used. Handles both CG and DG.

        Args:
            v: The test (weighting) function for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            The linear form.
        """

        # Source term
        L = dt * self.f[time_step] * v * ngs.dx

        # Dirichlet BCs for u, they only get added if using DG
        if self.DG:
            # Define the special DG functions.
            n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * self.dc[time_step] * (
                    alpha * g * v          # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - g * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    ) * self._ds(marker)

        # Neumann BCs for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker][time_step]
            L += dt * self.dc[time_step] * h * v * self._ds(marker)

        # Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker][time_step]
            L += -dt * self.dc[time_step] * r * q * v * self._ds(marker)

        return L
