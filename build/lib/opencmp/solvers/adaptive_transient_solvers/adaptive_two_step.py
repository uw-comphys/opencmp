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
from ...models import Model
from typing import Tuple, Type, List, Optional
from ...config_functions import ConfigParser
from ..time_integration_schemes import implicit_euler, crank_nicolson
from .base_adaptive_transient_multistep import BaseAdaptiveTransientMultiStepSolver


class AdaptiveTwoStep(BaseAdaptiveTransientMultiStepSolver):
    """
    An adaptive time-stepping solver that uses a two-step method.

    Each timestep is solved first with implicit Euler (first-order) and then with Crank-Nicolson (second-order). The
    local error is estimated from the L2 norm of the difference between the solutions. If the timestep is accepted the
    implicit Euler solution is kept since implicit Euler is unconditionally stable.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

        self.gfu_pred = self.model.construct_gfu()

    def reset_model(self) -> None:
        super().reset_model()

        self.gfu_pred = self.model.construct_gfu()

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu_pred)
        self.model.apply_dirichlet_bcs_to(self.gfu)

    def _assemble(self) -> None:
        for i in range(len(self.a_pred)):
            self.a_pred[i].Assemble()
            self.L_pred[i].Assemble()
        self._update_preconditioners(self.preconditioner_pred)

        for i in range(len(self.a_corr)):
            self.a_corr[i].Assemble()
            self.L_corr[i].Assemble()
        self._update_preconditioners(self.preconditioner_corr)

    def _create_linear_and_bilinear_forms(self) -> None:
        self.a_pred, self.L_pred = crank_nicolson(self.model, self.gfu_0_list, self.dt_param)
        self.a_corr, self.L_corr = implicit_euler(self.model, self.gfu_0_list, self.dt_param)

    def _create_preconditioners(self) -> None:
        self.preconditioner_pred = self.model.construct_preconditioners(self.a_pred)
        self.preconditioner_corr = self.model.construct_preconditioners(self.a_corr)

    def _update_preconditioners(self, precond_lst: List[Optional[Preconditioner]] = None) -> None:
        for preconditioner in precond_lst:
            if preconditioner is not None:
                preconditioner.Update()

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        self.model.solve_single_step(self.a_pred, self.L_pred, self.preconditioner_pred, self.gfu_pred)
        self.model.solve_single_step(self.a_corr, self.L_corr, self.preconditioner_corr, self.gfu)

    def _calculate_local_error(self) -> Tuple[List[float], List[float], List[str]]:
        # Include any variables specified by the model as included in local error.
        local_errors = []

        # Also get the gridfunction norms to use for the relative error tolerance.
        gfu_norms = []

        # Get the component names in the order that they were read
        comp_names = []

        if len(self.gfu.components) == 0:
            # Only one model variable to estimate local error with.
            local_errors.append(ngs.sqrt(ngs.Integrate((self.gfu - self.gfu_pred) ** 2, self.model.mesh)))
            gfu_norms.append(ngs.sqrt(ngs.Integrate(self.gfu ** 2, self.model.mesh)))
            comp_names.append(list(self.model.model_components.keys())[0])
        else:
            # Include any variables specified by the model as included in local error.
            for comp_name, use in self.model.model_local_error_components.items():
                if use:
                    comp_index = self.model.model_components[comp_name]
                    local_errors.append(ngs.sqrt(ngs.Integrate((self.gfu.components[comp_index] -
                                                                self.gfu_pred.components[comp_index]) ** 2,
                                                               self.model.mesh)))
                    gfu_norms.append(ngs.sqrt(ngs.Integrate(self.gfu.components[comp_index] ** 2, self.model.mesh)))
                    comp_names.append(comp_name)

        return local_errors, gfu_norms, comp_names
