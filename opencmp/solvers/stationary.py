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
from ngsolve import Parameter, Preconditioner
from typing import List, Optional
from .base_solver import Solver
from typing import Tuple

"""
Module for the stationary solver class.
"""


class StationarySolver(Solver):
    """
    Stationary solver.
    """

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu)

    def _create_linear_and_bilinear_forms(self) -> None:
        U, V = self.model.get_trial_and_test_functions()

        # Bilinear form
        self.a = []
        a_coeff_terms = self.model.construct_bilinear_time_coefficient(U, V, Parameter(1.0), 0)
        a_ode_terms = self.model.construct_bilinear_time_ODE(U, V)

        for i in range(self.model.num_weak_forms):
            a = ngs.BilinearForm(self.model.fes)
            a += a_coeff_terms[i]
            a += a_ode_terms[i]
            self.a.append(a)

        # Linear form
        # NOTE: No IMEX terms since IMEX should only be used with transient solves.
        self.L = []
        L_terms = self.model.construct_linear(V, None, Parameter(1.0), 0)

        for i in range(self.model.num_weak_forms):
            L = ngs.LinearForm(self.model.fes)
            L += L_terms[i]
            self.L.append(L)

    def _create_preconditioners(self) -> None:
        self.preconditioners = self.model.construct_preconditioners(self.a)

    def _load_and_apply_initial_conditions(self) -> None:
        self.gfu_0_list: List[ngs.GridFunction] = [self.model.construct_gfu()]

    def _log_timestep(self, accepted: bool, error_abs: float, error_rel: float, component: str) -> None:
        # Print nothing since it's a single iteration
        pass

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:

        # performs a single linear solve in order to determine the current iterate value
        self.model.solve_single_step(self.a, self.L, self.preconditioners, self.gfu)

    def _startup(self) -> None:
        # Not applicable.
        pass

    def _update_time_step(self) -> Tuple[bool, float, float, str]:
        # Do nothing since it's a stationary solver
        return True, -1.0, -1.0, ''
