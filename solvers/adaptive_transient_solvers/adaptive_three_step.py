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
from models import Model
from typing import Tuple, Type, List, Optional
from config_functions import ConfigParser
from time_integration_schemes import implicit_euler
from .base_adaptive_transient_RK import BaseAdaptiveTransientRKSolver

class AdaptiveThreeStep(BaseAdaptiveTransientRKSolver):
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
        self.model.apply_dirichlet_bcs_to(self.gfu_long, 0)
        self.model.apply_dirichlet_bcs_to(self.gfu_short, 1)
        self.model.apply_dirichlet_bcs_to(self.gfu, 0)

    def _assemble(self) -> None:
        self.a_long.Assemble()
        self.L_long.Assemble()

    def _create_linear_and_bilinear_forms(self) -> None:
        self.a_long, self.L_long = implicit_euler(self.model, self.gfu_0_list, self.dt_param, 0)
        self.a_short, self.L_short = implicit_euler(self.model, self.gfu_0_list, self.dt_param, 1)
        self.a, self.L = implicit_euler(self.model, [self.gfu_short], [self.dt_param[1]], 0)

    def _create_preconditioner(self) -> None:
        self.preconditioner_long = self.model.construct_preconditioner(self.a_long)
        self.preconditioner_short = self.model.construct_preconditioner(self.a_short)
        self.preconditioner = self.model.construct_preconditioner(self.a)

    def _update_preconditioner(self, precond: Optional[Preconditioner] = None) -> None:
        if precond is not None:
            precond.Update()

    def _re_assemble(self) -> None:
        self._assemble()
        self._update_preconditioner(self.preconditioner_long)

    def _single_solve(self) -> None:
        # Single solve for the full time step.
        self.model.single_iteration(self.a_long, self.L_long, self.preconditioner_long, self.gfu_long, 0)

        # Update the linearization terms back to their t^n values.
        self.model.update_linearization_terms(self.gfu_0_list[0])

        # Single solve for the first half of the time step.
        self.a_short.Assemble()
        self.L_short.Assemble()
        self._update_preconditioner(self.preconditioner_short)
        self.model.single_iteration(self.a_short, self.L_short, self.preconditioner_short, self.gfu_short, 1)

        # Update the model component values at t^n+1/2 now that they have been solved for.
        # The linearization terms do not need to be updated since they are now for t^n+1/2 as expected.
        self.model.update_model_variables(self.gfu_short, time_step=1)

        # Single solve for the second half of the time step.
        self.a.Assemble()
        self.L.Assemble()
        self._update_preconditioner(self.preconditioner)
        self.model.single_iteration(self.a, self.L, self.preconditioner, self.gfu, 0)

    def _calculate_local_error(self) -> Tuple[List[float], List[float], List[str]]:
        # Include any variables specified by the model as included in local error.
        local_errors = []

        # Also get the gridfunction norms to use for the relative error tolerance.
        gfu_norms = []

        # Get the component names in the order that they were read
        comp_names = []

        if len(self.gfu.components) == 0:
            # Only one model variable to estimate local error with.
            local_errors.append(ngs.sqrt(ngs.Integrate((self.gfu - self.gfu_long) ** 2, self.model.mesh)))
            gfu_norms.append(ngs.sqrt(ngs.Integrate(self.gfu ** 2, self.model.mesh)))
            comp_names.append(list(self.model.model_components.keys())[0])
        else:
            # Include any variables specified by the model as included in local error.
            for comp_name, use in self.model.model_local_error_components.items():
                if use:
                    comp_index = self.model.model_components[comp_name]
                    local_errors.append(ngs.sqrt(ngs.Integrate((self.gfu.components[comp_index] -
                                                                self.gfu_long.components[comp_index]) ** 2,
                                                               self.model.mesh)))
                    gfu_norms.append(ngs.sqrt(ngs.Integrate(self.gfu.components[comp_index] ** 2, self.model.mesh)))
                    comp_names.append(comp_name)

        return local_errors, gfu_norms, comp_names
