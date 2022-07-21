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
from .stokes import Stokes
from ngsolve.comp import ProxyFunction
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm, Preconditioner
from typing import List, Union, Optional


class StokesDIM(Stokes):
    """
    A single phase Stokes model with the Diffuse Interface Method.
    """

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test function for velocity
        u = U[0]
        v = V[0]

        a = dt * (
                self.kv[time_step] * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
        ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u to zero where phi is zero.
        # If DIM rigid body motion is being used force u to the velocity of the rigid body instead.
        a += dt * alpha * u * v * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        # Penalty terms for conformal Dirichlet BCs
        if self.DG and self.dirichlet_names.get('u', None) is not None:
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

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                            dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure
        u, p = U[0], U[1]
        v, q = V[0], V[1]

        a = dt * (
                - ngs.div(u) * q  # Conservation of mass
                - ngs.div(v) * p  # Pressure
                - 1e-10 * p * q  # Stabilization term.
        ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force grad(p) to zero where phi is zero. If rigid body motion is being used ignore this.
        if not self.DIM_solver.rigid_body_motion:
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

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                         dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:

        # Separate out the test function for velocity.
        v = V[0]

        # Define the base linear form
        L = dt * v * self.f['u'][time_step] * self.DIM_solver.phi_gfu * ngs.dx

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Force u to zero where phi is zero. If DIM rigid body motion is being used force u to the velocity of the
        # rigid body instead.
        if self.DIM_solver.rigid_body_motion:
            for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
                g = self.DIM_BC['dirichlet']['u'][marker][time_step]
                L += dt * alpha * g * v * (1.0 - self.DIM_solver.phi_gfu) * self.DIM_solver.mask_gfu_dict[
                    marker] * ngs.dx

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
        for marker in self.BC.get('stress', {}).get('u', {}):
            h = self.BC['stress']['u'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return [L]

