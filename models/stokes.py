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
from helpers.dg import jump, grad_avg
from models import INS
from ngsolve.comp import ProxyFunction
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm, Preconditioner
from typing import Tuple, List, Union, Optional
from config_functions import ConfigParser


class Stokes(INS):
    """
    A single phase Stokes model.
    """
    def __init__(self, config: ConfigParser, t_param: List[Parameter]) -> None:
        # Specify information about the model components
        # NOTE: These MUST be set before calling super(), since it's needed in superclass' __init__
        self.model_components               = {'u': 0,      'p': 1}
        self.model_local_error_components   = {'u': True,   'p': True}
        self.time_derivative_components     = {'u': True,   'p': False}

        # Pre-define which BCs are accepted for this model, all others are thrown out.
        self.BC_init = {'dirichlet':    {},
                        'stress':       {},
                        'pinned':       {}}

        # Bypass INS' __init__ and directly call Model's __init__
        super(INS, self).__init__(config, t_param)

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        # Split up the trial (weighting) and test functions
        u, p = U
        v, q = V

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_ODE_DIM(u, v, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_ODE_no_DIM(u, v, dt, time_step)

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        # Split up the trial (weighting) and test functions
        u, p = U
        v, q = V

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_time_coefficient_DIM(u, p, v, q, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_time_coefficient_no_DIM(u, p, v, q, dt, time_step)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                         dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:
        # Split up the test functions
        v, q = V

        if self.DIM:
            # Use the diffuse interface method.
            L = self._linear_DIM(v, dt, time_step)
        else:
            # Solve on a standard conformal mesh.
            L = self._Linear_no_DIM(v, dt, time_step)

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                                dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:
        # The Stokes equations are linear and have no need for IMEX schemes.
        pass

    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        # Define the test and trial functions.
        u, p = self.fes.TrialFunction()
        v, q = self.fes.TestFunction()

        return [u, p], [v, q]

    def single_iteration(self, a: BilinearForm, L: LinearForm, precond: Preconditioner, gfu: GridFunction,
                         time_step: int = 0) -> None:

        self.construct_and_run_solver(a, L, precond, gfu)

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################

    def _bilinear_time_ODE_DIM(self, u: Union[ProxyFunction, GridFunction], v: ProxyFunction, dt: Parameter,
                               time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables with time derivatives.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        a = dt * (
            self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u to zero where phi is zero.
        a += dt * alpha * u * v * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += -dt * self.kv[time_step] * (
                    ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                    - alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += dt * self.kv[time_step] * (
                ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, self.DIM_solver.grad_phi_gfu))
                + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, self.DIM_solver.grad_phi_gfu))
                + alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # TODO: Add non-Dirichlet DIM BCs.

        return a

    def _bilinear_time_coefficient_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction,
                                       dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables without time derivatives.
        """

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        a = dt * (
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            - 1e-10 * p * q   # Stabilization term.
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force grad(p) to zero where phi is zero.
        a += -dt * p * ngs.div(v) * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a += -dt * self.kv[time_step] * (
                ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                + ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                - alpha * ngs.InnerProduct(jump_u, jump_v)  # Term for u+=u- on 𝚪_I from ∇u^
                ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

        return a

    def _bilinear_time_ODE_no_DIM(self, u: Union[ProxyFunction, GridFunction], v: ProxyFunction, dt: Parameter,
                                  time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is not being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables with time derivatives.
        """

        a = dt * (
            self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            ) * ngs.dx

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Bulk of Bilinear form
        if self.DG:
            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += -dt * self.kv[time_step] * (
                    ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                    - alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        return a

    def _bilinear_time_coefficient_no_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction,
                                  dt: Parameter, time_step: int) -> BilinearForm:
        """
        Bilinear form when the diffuse interface method is not being used. Handles both CG and DG.
        This is the portion of the bilinear form for model variables without time derivatives.
        """

        a = dt * (
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            - 1e-10 * p * q   # Stabilization term
            ) * ngs.dx

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Bulk of Bilinear form
        if self.DG:
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a += -dt * self.kv[time_step] * (
                ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                + ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                - alpha * ngs.InnerProduct(jump_u, jump_v)  # Term for u+=u- on 𝚪_I from ∇u^
                ) * ngs.dx(skeleton=True)

        return a

    def _linear_DIM(self, v: ProxyFunction, dt: Parameter, time_step: int) -> LinearForm:
        """ Linear form when the diffuse interface method is being used. Handles both CG and DG. """

        # Define the base linear form
        L = dt * v * self.f[time_step] * self.DIM_solver.phi_gfu * ngs.dx

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        if self.DG:
            # Conformal Dirichlet BCs for u
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * self.kv[time_step] * (
                    alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    ) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            g = self.DIM_BC['dirichlet']['u'][marker][time_step]
            L += dt * self.kv[time_step] * (
                alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
                + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, self.DIM_solver.grad_phi_gfu))
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Stress BCs
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return L

    def _Linear_no_DIM(self, v: ProxyFunction, dt: Parameter, time_step: int) -> LinearForm:
        """ Linear form when the diffuse interface method is not being used. Handles both CG and DG. """

        # Define the base linear form
        L = dt * v * self.f[time_step] * ngs.dx

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        if self.DG:
            # Dirichlet BCs for u
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * self.kv[time_step] * (
                    alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    ) * self._ds(marker)

        # Stress BCs
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        return L
