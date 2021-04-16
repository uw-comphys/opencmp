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
import math
# TODO: generalize so we don't have to import each one individually
from models import Model
from typing import Tuple, Type, List, Union
from config_functions import ConfigParser
from .transient import TransientSolver
from abc import ABC, abstractmethod
from helpers.saving import SolutionFileSaver
import sys

"""
Module for the adaptive transient solver class.
"""

class AdaptiveTransientSolver(TransientSolver, ABC):
    """
    A transient solver with adaptive time-stepping.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

    def _update_time_step(self) -> Tuple[bool, Union[float, str]]:
        dt_min_allowed = self.dt_range[0]
        dt_max_allowed = self.dt_range[1]

        # Get the local error and the norm of the solution gridfunction.
        local_error, gfu_norm = self._calculate_local_error()

        # Calculate the next timestep based on the relative local error and multiply by 0.9 for a safety margin.
        all_dt_from_local_error = [0.9 * self.dt_param[0].Get() * min(max(math.sqrt((self.dt_abs_tol
                                                                                     + self.dt_rel_tol * gfu_norm[i])
                                                                                    / local_error[i]), 0.3), 2.0)
                                   for i in range(len(local_error))]
        # Pick the smallest of the timestep values.
        dt_from_local_error = min(all_dt_from_local_error)

        # Ensure dt_from_local_error is not smaller than the specified minimum.
        # This is checked to ensure that dt does not decrease to ~1e-64 or lower when the solver can't take a step
        if dt_from_local_error < dt_min_allowed:
            # Save the current solution before ending the run.
            if self.save_to_file:
                self.saver.save(self.gfu, self.t_param.Get())
            else:
                tmp_saver = SolutionFileSaver(self.model, quiet=True)
                tmp_saver.save(self.gfu, self.t_param.Get())

            sys.exit('At t = {0} further time steps must be smaller than the minimum time step. Saving current '
                     'solution to file and ending the run. Suggest rerunning with a time step of {1} s.'
                     .format(self.t_param.Get(), dt_from_local_error))

        dt_new = min([dt_from_local_error, self._dt_for_next_time_to_hit(), dt_max_allowed])

        # Accept the time step if all local errors are less than the absolute and relative tolerance.
        accept_timestep = all([local_error[i] <= self.dt_abs_tol + self.dt_rel_tol * gfu_norm[i] for i in range(len(local_error))])

        if accept_timestep:
            # Keep the solution and move to the next time step with the new dt.
            #
            # Update all previous timestep solutions and dt values.
            for i in range(1, self.scheme_order - 1):
                self.gfu_0_list[-i].vec.data = self.gfu_0_list[-(i + 1)].vec
                self.dt_param[-i].Set(self.dt_param[-(i + 1)].Get())

            # Update data in previous timestep with new solution
            self.gfu_0_list[0].vec.data = self.gfu.vec

            # Update the new dt_param value.
            self.dt_param[0].Set(dt_new)

        else:
            # Repeat the time step with the new dt.
            self.t_param.Set(self.t_param.Get() - self.dt_param[0].Get())
            self.dt_param[0].Set(dt_new)

        # Return whether to accept the timestep and the maximum relative local error.
        return accept_timestep, max([local_error[i] / gfu_norm[i] for i in range(len(local_error))])

    @abstractmethod
    def _calculate_local_error(self) -> Tuple[Union[str, List], List]:
        """
        Function to calculate the local error in the time step's solution and the norm of the time step's solution.

        Returns:
            local_error: List of local error for each model variable as specified by model_local_error_components.
            gfu_norm: List of solution norm for each model variable as specified by model_local_error_components.
        """
