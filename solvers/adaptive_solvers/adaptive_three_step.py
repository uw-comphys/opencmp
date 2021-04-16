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
from models import Model
from typing import Tuple, Type, List, Union
from config_functions import ConfigParser
from time_integration_schemes import implicit_euler
from ..adaptive_transient import AdaptiveTransientSolver


class AdaptiveThreeStep(AdaptiveTransientSolver):
    """
    A transient solver with adaptive time-stepping that uses a three step method.

    Each timestep is solved first with one implicit Euler solve with dt and then with two implicit Euler solves with
    dt/2. Local error is estimated from the L2 norm of the difference between the two final timestep solutions. If the
    timestep is accepted the solution using dt/2 is kept as the solution at the timestep as that should be the most
    accurate solution.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

        self.gfu_long = self.model.construct_gfu()
        self.gfu_short = self.model.construct_gfu()

    def reset_model(self) -> None:
        super().reset_model()

        self.gfu_long = self.model.construct_gfu()
        self.gfu_short = self.model.construct_gfu()

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu_long)
        self.model.apply_dirichlet_bcs_to(self.gfu_short)
        self.model.apply_dirichlet_bcs_to(self.gfu)

    def _assemble(self) -> None:
        self.a_long.Assemble()
        self.L_long.Assemble()

        self.a_short.Assemble()
        self.L_short.Assemble()

    def _create_linear_and_bilinear_forms(self) -> None:
        self.dt_param_half = [0.5 * dt for dt in self.dt_param]
        self.a_long, self.L_long = implicit_euler(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        self.a_short, self.L_short = implicit_euler(self.model, self.gfu_0_list, self.U, self.V, self.dt_param_half)
        self.a, self.L = implicit_euler(self.model, [self.gfu_short], self.U, self.V, self.dt_param_half)

    def _create_preconditioner(self) -> None:
        self.preconditioner_long = self.model.construct_preconditioner(self.a_long)
        self.preconditioner_short = self.model.construct_preconditioner(self.a_short)
        self.preconditioner = self.model.construct_preconditioner(self.a)

    def _re_assemble(self) -> None:
        self._assemble()
        self.preconditioner_long.Update()
        self.preconditioner_short.Update()

    def _single_solve(self) -> None:
        self.model.single_iteration(self.a_long, self.L_long, self.preconditioner_long, self.gfu_long)
        self.model.single_iteration(self.a_short, self.L_short, self.preconditioner_short, self.gfu_short)

        # Reassemble a and L after gfu_short has been solved.
        self.a.Assemble()
        self.L.Assemble()
        self.preconditioner.Update()

        self.model.single_iteration(self.a, self.L, self.preconditioner, self.gfu)

    def _calculate_local_error(self) -> Tuple[List[float], List[float]]:
        if len(self.gfu.components) == 0:
            # Only one model variable to estimate local error with.
            local_errors = [ngs.sqrt(ngs.Integrate((self.gfu - self.gfu_long) ** 2, self.model.mesh))]

            # Also get the gridfunction norm to use for the relative error tolerance.
            gfu_norms = [ngs.sqrt(ngs.Integrate(self.gfu ** 2, self.model.mesh))]

        else:
            # Include any variables specified by the model as included in local error.
            local_errors = []

            # Also get the gridfunction norms to use for the relative error tolerance.
            gfu_norms = []

            for var, use in self.model.model_local_error_components.items():
                if use:
                    component = self.model.model_components[var]
                    local_errors.append(ngs.sqrt(ngs.Integrate((self.gfu.components[component] -
                                                               self.gfu_long.components[component]) ** 2,
                                                              self.model.mesh)))
                    gfu_norms.append(ngs.sqrt(ngs.Integrate(self.gfu.components[component] ** 2, self.model.mesh)))

        return local_errors, gfu_norms
