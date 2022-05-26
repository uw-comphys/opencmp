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

from . import Poisson
from typing import List, Union, Optional
from ngsolve.comp import ProxyFunction
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm


class PoissonDIM(Poisson):
    """
    A poisson model with the Diffuse Interface Method
    """

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        # Split up the trial (weighting) and test functions
        u = U[0]
        v = V[0]

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

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:
        # Split up the trial (weighting) and test functions
        u = U[0]
        v = V[0]

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
                    - avg_grad_u * jump_v * n  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                    + alpha * jump_u * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
            ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

        else:
            a = ngs.CoefficientFunction(0.0) * ngs.dx

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:
        # V is a list of length 1 for Poisson, get just the first element
        v = V[0]

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Source term
        L = dt * self.f[time_step] * v * self.DIM_solver.phi_gfu * ngs.dx

        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                # Penalty terms for conformal Dirichlet BCs
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * self.dc[time_step] * (
                        alpha * g * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
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

        return [L]