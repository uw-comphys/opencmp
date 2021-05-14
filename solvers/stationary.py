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
from ngsolve import Preconditioner
from typing import Optional
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

    def _assemble(self) -> None:
        self.a.Assemble()
        self.L.Assemble()
        self._update_preconditioner()

    def _create_linear_and_bilinear_forms(self) -> None:
        # TODO: Add something to handle a model with multiple coupled solves.
        U, V = self.model.get_trial_and_test_functions()
        
        self.a = ngs.BilinearForm(self.model.fes)
        self.a += self.model.construct_bilinear_time_coefficient(U, V)[0]
        self.a += self.model.construct_bilinear_time_ODE(U, V)[0]

        self.L = ngs.LinearForm(self.model.fes)
        self.L += self.model.construct_linear(V)[0]
        # No IMEX terms since IMEX should only be used with transient solves.

    def _create_preconditioner(self) -> None:
        self.preconditioner = self.model.construct_preconditioner(self.a)

    def _update_preconditioner(self, precond: Optional[Preconditioner] = None) -> None:
        if self.preconditioner is not None:
            self.preconditioner.Update()

    def _load_and_apply_initial_conditions(self) -> None:
        # Nothing to do since it's a stationary solve
        pass

    def _log_timestep(self, accepted: bool, error_abs: float, error_rel: float, component: str) -> None:
        # Print nothing since it's a single iteration
        pass

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        self.model.single_iteration(self.a, self.L, self.preconditioner, self.gfu)

    def _startup(self) -> None:
        # Not applicable.
        pass

    def _update_time_step(self) -> Tuple[bool, float, float, str]:
        # Do nothing since it's a stationary solver
        return True, -1.0, -1.0, ''
