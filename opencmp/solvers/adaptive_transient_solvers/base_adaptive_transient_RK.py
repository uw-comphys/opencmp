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

import math
# TODO: generalize so we don't have to import each one individually
from ...models import Model
from typing import Tuple, Type, List
from ...config_functions import ConfigParser
from ...solvers import TransientRKSolver
from abc import ABC, abstractmethod
from ...helpers.saving import SolutionFileSaver
import sys

"""
Module for the Runge Kutta adaptive transient solver class.
"""


class BaseAdaptiveTransientRKSolver(TransientRKSolver, ABC):
    """
    A transient Runge Kutta solver with adaptive time-stepping.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

    def _update_time_step(self) -> Tuple[bool, float, float, str]:
        dt_min_allowed = self.dt_range[0]
        dt_max_allowed = self.dt_range[1]

        # Get the local error and the norm of the solution gridfunction.
        local_error, gfu_norm, comp_names = self._calculate_local_error()

        # Create a list of all of the relative errors
        local_error_relative = local_error.copy()

        # Ensure we don't divide by 0
        for i in range(len(local_error)):
            try:
                local_error_relative[i] /= gfu_norm[i]
            except ZeroDivisionError:
                # Leave it as is
                pass

            if local_error[i] == 0:
                local_error[i] = 1e-10

            if gfu_norm[i] == 0:
                gfu_norm[i] = 1e-10

        # Calculate the next timestep based on the relative local error and multiply by 0.9 for a safety margin.
        all_dt_from_local_error = [0.9 * self.dt_param[0].Get() * min(max(math.sqrt((self.dt_abs_tol
                                                                                     + self.dt_rel_tol * gfu_norm[i])
                                                                                    / local_error[i]), 0.3), 2.0)
                                   for i in range(len(local_error))]
        # Pick the smallest of the timestep values.
        dt_from_local_error = min(all_dt_from_local_error)

        # Ensure dt_from_local_error is not smaller than the specified minimum.
        # This is checked to ensure that dt does not decrease to ~1e-64 or lower when the solver can't take a step.
        if dt_from_local_error < dt_min_allowed:
            # Save the current solution before ending the run.
            if self.save_to_file and self.saver is not None:
                self.saver.save(self.gfu, self.t_param[0].Get())
            else:
                tmp_saver = SolutionFileSaver(self.model, quiet=True)
                tmp_saver.save(self.gfu, self.t_param[0].Get())

            print('At t = {0} further time steps must be smaller than the minimum time step. Saving current'
                  'solution to file and ending the run. Suggest rerunning with a time step of {1} s.'
                  .format(self.t_param[0].Get(), dt_from_local_error))

            sys.exit(1337)

        dt_new = min([dt_from_local_error, self._dt_for_next_time_to_hit(), dt_max_allowed])

        # Accept the time step if all local errors are less than the absolute and relative tolerance.
        accept_timestep = all([local_error[i] <= self.dt_abs_tol + self.dt_rel_tol * gfu_norm[i]
                               for i in range(len(local_error))])

        if accept_timestep:
            # Keep the solution and move to the next time step with the new dt.
            #
            # Set all intermediate step solutions to the current solution. Set all dt values to the next dt (may vary to
            # hit a save point).
            self.dt_param[-1].Set(self.dt_param[0].Get())
            for i in range(self.scheme_order):
                self.gfu_0_list[i].vec.data = self.gfu.vec
                self.dt_param[i].Set(dt_new * self.scheme_dt_coef[i])

            # Update the values of the model variables based on the newly accepted timestep
            # and re-parse the model functions as necessary.
            self.model.update_model_variables(self.gfu, time_step=self.scheme_order)

            # Update the model component values for all the intermediate time steps to be the trial functions.
            for i in range(1, self.scheme_order):
                self.model.update_model_variables(self.U, time_step=i)

            # Reset self.step to one.
            self.step = 1

        else:
            # Repeat the time step with the new dt.
            #
            # Update the linearization terms back to their starting values for the time step that will be rerun.
            self.model.update_linearization_terms(self.gfu_0_list[0])

            # Update the values of the model variables based on the last accepted timestep
            # and re-parse the model functions as necessary.
            if self.scheme == 'adaptive three step':
                self.model.update_model_variables(self.gfu_0_list[0])
            else:
                self.model.update_model_variables(self.gfu_0_list[0], time_step=self.scheme_order)

            # Set all intermediate step solutions to the previous solution.
            for i in range(self.scheme_order - 1):
                self.gfu_0_list[i].vec.data = self.gfu_0_list[-1].vec

            # Reset the current time value to the previous time step.
            self.t_param[0].Set(self.t_param[0].Get() - self.dt_param[0].Get())

            # Set all dt values to the new dt except for the last dt value (keep as the dt for the previous time step).
            if self.scheme == 'adaptive three step':
                dt_ref = self.dt_param[1].Get()
            else:
                dt_ref = dt_new
            self.dt_param[-1].Set(self.dt_param[0].Get())
            for i in range(self.scheme_order):
                self.dt_param[i].Set(dt_ref * self.scheme_dt_coef[i])

            # Now adjust all intermediate time values to the correct value for the previous time step based on the new
            # dt.
            self.t_param[-1].Set(self.t_param[0].Get() - self.dt_param[0].Get())
            for i in range(1, self.scheme_order):
                self.t_param[i].Set(self.t_param[0].Get() - self.dt_param[i].Get())

            # Reset self.step to one.
            self.step = 1

        # The largest absolute error values
        max_abs = max(local_error)

        # The index for the largest value
        i = local_error.index(max_abs)

        # The largest relative error values
        max_rel = local_error_relative[i]

        # The name of the component associated with the largest error
        component = comp_names[i]

        # Return whether to accept the timestep and the maximum relative local error.
        return accept_timestep, max_abs, max_rel, component

    @abstractmethod
    def _calculate_local_error(self) -> Tuple[List[float], List[float], List[str]]:
        """
        Function to calculate the local error in the time step's solution and the norm of the time step's solution.

        Returns:
            Tuple[List[float], List[float], List[str]]
                - local_error: List of local error for each model variable as specified by model_local_error_components.
                - gfu_norm: List of solution norm for each model variable as specified by model_local_error_components.
                - component_names: List of names for the components that were tested, in the order that they were.
        """
