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
from opencmp.solvers.time_integration_schemes import adaptive_IMEX_pred
from .base_adaptive_transient_multistep import BaseAdaptiveTransientMultiStepSolver


class AdaptiveIMEX(BaseAdaptiveTransientMultiStepSolver):
    """
    A transient solver with IMEX-based adaptive time-stepping.

    Each timestep is solved with a first-order IMEX method then updated with a second order explicit solve. Two
    estimates of local error are made. The solution accepted as the timestep's solution depends on which local error
    estimate meets the specified tolerance.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

        self.gfu_pred = self.model.construct_gfu()
        self.gfu_corr = self.model.construct_gfu()

    def reset_model(self) -> None:
        super().reset_model()

        self.gfu_pred = self.model.construct_gfu()
        self.gfu_corr = self.model.construct_gfu()

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu_pred)

    def _assemble(self) -> None:
        for i in range(len(self.a)):
            self.a[i].Assemble()
            self.L[i].Assemble()

    def _create_linear_and_bilinear_forms(self) -> None:
        self.a_pred, self.L_pred = adaptive_IMEX_pred(self.model, self.gfu_0_list, self.dt_param)

    def _create_preconditioners(self) -> None:
        self.preconditioner_pred = self.model.construct_preconditioners(self.a_pred)

    def _update_preconditioners(self, precond_lst: List[Optional[Preconditioner]] = None) -> None:
        for preconditioner in precond_lst:
            if preconditioner is not None:
                preconditioner.Update()

    def _re_assemble(self) -> None:
        self._assemble()
        self._update_preconditioners(self.preconditioner_pred)

    def _single_solve(self) -> None:
        self.model.solve_single_step(self.a_pred, self.L_pred, self.preconditioner_pred, self.gfu_pred)

        # Correction only happens to the velocity.
        component = self.model.model_components['u']

        # Operators specific to this time integration scheme.
        w = self.dt_param[0] / self.dt_param[1]
        E = (1.0 + w) * self.gfu_0_list[0].components[component] - w * self.gfu_0_list[1].components[component]

        # Explicit evaluation of the corrector.
        corr_expr = self.gfu_pred.components[component] \
                    - w / (2.0 * w + 1.0) * (self.gfu_pred.components[component] - E)
        self.gfu_corr.components[component].Set(corr_expr)

    def _calculate_local_error(self) -> Tuple[List[float], List[float], List[str]]:
        # Use the model component corresponding to velocity.
        component = self.model.model_components['u']

        # Operators specific to this time integration scheme.
        w_0 = self.dt_param[0] / self.dt_param[1]
        w_00 = self.dt_param[1] / self.dt_param[2]

        # The two error expressions.
        err_1 = ngs.sqrt(ngs.Integrate((self.gfu_pred.components[component] - self.gfu_corr.components[component]) ** 2,
                                       self.model.mesh))

        err_2_expr = w_00 * w_0 * (1.0 + w_0) / (1.0 + 2.0 * w_0 + w_00 * (1.0 + 4.0 * w_0 + 3.0 * w_0 * w_0)) \
                     * (self.gfu_corr.components[component]
                        - (1.0 + w_0) * (1.0 + w_00 * (1.0 + w_0)) * self.gfu_0_list[0].components[component] / (1.0 + w_00)
                        + w_0 * (1.0 + w_00 * (1.0 + w_0)) * self.gfu_0_list[1].components[component]
                        - w_00 * w_00 * w_0 * (1.0 + w_0) * self.gfu_0_list[2].components[component] / (1.0 + w_00))
        err_2 = ngs.sqrt(ngs.Integrate(err_2_expr ** 2, self.model.mesh))

        # Keep the larger of the errors that meet the tolerance.
        # TODO: This only accounts for absolute tolerance, following the paper.
        #       Possibly should also include relative tolerance.
        if (err_1 <= self.dt_abs_tol) and (err_2 <= self.dt_abs_tol):
            if err_1 > err_2:
                # Keep the solution from the predictor (if the timestep is accepted).
                self.gfu.vec.data = self.gfu_pred.vec
                local_error = [err_1]
            else:
                # Keep the solution from the corrector (if the timestep is accepted).
                self.gfu.components[0].vec.data = self.gfu_corr.components[0].vec
                self.gfu.components[1].vec.data = self.gfu_pred.components[1].vec
                local_error = [err_2]
        elif err_1 <= self.dt_abs_tol:
            self.gfu.vec.data = self.gfu_pred.vec
            local_error = [err_1]
        elif err_2 <= self.dt_abs_tol:
            # Keep the solution from the corrector (if the timestep is accepted).
            self.gfu.components[0].vec.data = self.gfu_corr.components[0].vec
            self.gfu.components[1].vec.data = self.gfu_pred.components[1].vec
            local_error = [err_2]
        else:
            # Neither error will be acceptable. Output the error closest to the tolerance.
            self.gfu.vec.data = self.gfu_pred.vec
            local_error = [min(err_1, err_2)]

        # Make sure the correct gridfunction is being used to get the norm for the relative error tolerance. It
        # should be the gridfunction corresponding to the maximum local error.
        gfu_norm = [ngs.sqrt(ngs.Integrate(self.gfu.components[component] ** 2, self.model.mesh))]

        return local_error, gfu_norm, ['']
