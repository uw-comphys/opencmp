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
from typing import Dict, List, Tuple, Union, Optional

import ngsolve as ngs
from ngsolve.comp import FESpace, ProxyFunction
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm, Preconditioner

from ..helpers.ngsolve_ import get_special_functions
from ..helpers.dg import jump, grad_avg
from . import Model


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
        return ['dirichlet', 'neumann', 'robin']

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
                    - avg_grad_u * n * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                    + alpha * jump_u * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
            ) * ngs.dx(skeleton=True)

        else:
            a = ngs.CoefficientFunction(0.0) * ngs.dx

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:
        # V is a list of length 1 for Poisson, get just the first element
        v = V[0]

        # Source term
        L = dt * self.f[time_step] * v * ngs.dx

        # Dirichlet BCs for u, they only get added if using DG
        if self.DG:
            # Define the special DG functions.
            n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * self.dc[time_step] * (
                        alpha * g * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
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

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:
        # The Poisson equation is linear and has no need for IMEX schemes.
        pass

    def solve_single_step(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                         precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:

        self.linearized_solve(a_lst[0], L_lst[0], precond_lst[0], gfu)
