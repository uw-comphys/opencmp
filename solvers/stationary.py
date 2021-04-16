"""
Copyright 2021 the authors (see AUTHORS file for full list)

This file is part of OpenCMP.

OpenCMP is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

OpenCMP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with OpenCMP.  If not, see <https://www.gnu.org/licenses/>.
"""

import ngsolve as ngs
from .base_solver import Solver

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
        self.preconditioner.Update()

    def _create_linear_and_bilinear_forms(self) -> None:
        self.a = ngs.BilinearForm(self.model.fes)
        self.a += self.model.construct_bilinear(self.U, self.V)

        self.L = ngs.LinearForm(self.model.fes)
        self.L += self.model.construct_linear(self.V)

    def _create_preconditioner(self) -> None:
        self.preconditioner = self.model.construct_preconditioner(self.a)

    def _load_and_apply_initial_conditions(self) -> None:
        # Nothing to do since it's a stationary solve
        pass

    def _log_timestep(self, accepted: bool, error: float) -> None:
        # Print nothing since it's a single iteration
        pass

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        self.model.single_iteration(self.a, self.L, self.preconditioner, self.gfu)

    def _startup(self) -> None:
        # Not applicable.
        pass

    def _update_time_step(self) -> bool:
        # Do nothing since it's a stationary solver
        return True, -1.0
