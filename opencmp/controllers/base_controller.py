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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from ngsolve import CoefficientFunction, exp, GridFunction, Parameter
from ..config_functions import ControllerFunctions
from ..models import Model


class Controller(ABC):
    """
    Base class which other controllers will subclass
    """

    def __init__(self, t_params: List[Parameter], model: Model, config_rel_path: str, import_dir: str) -> None:
        """
        Initializer for the controller classes.

        Args:
            t_params: Parameter representing the current time
            model: The model that the control is being applied to
            config_rel_path: The filename, and relative path, for the config file for this controller
            import_dir: The path to the main run directory containing the file from which to import any Python functions.
        """
        # The time of the simulation
        self.t_params = t_params

        # Save the model
        self.model = model

        # Config parameter loader
        self.config_func = ControllerFunctions(config_rel_path, import_dir, self.model.mesh, self.t_params)

        # Time step between each control action
        # TODO: How to also non-dimensionalize this?
        self.dt_control = self.config_func.config.get_item(['PARAMETERS', 'dt_control'], float)

        # Time constant for the dynamics of the control action
        self.tau: float = self.config_func.config.get_item(['PARAMETERS', 't_ramp'], float)

        # Check that tau <= dt_control. Alert the user if it's not
        # TODO: Is this warning still needed?
        if self.tau > self.dt_control:
            print('WARNING: tau is higher than dt_control, this was likely not intentional')

        # The time at which the next control action will take place
        self.t_next_action = self.t_params[0].Get()

        # Load info for manipulated variables
        # [[type_1, var_1, loc_1],[...],...]
        self.vars_manipulate = self.config_func.get_manipulated_variables()

        # Load info for control variables info
        # [[name_1, pos_1, val_1, index_1],[...],...]
        self.vars_control = self.config_func.get_control_variables()

    def _apply_dynamics_equation(self, next_control_action: float, prev_control_action: float, t_param: Parameter)\
            -> CoefficientFunction:
        """
        Function add dynamics to the transition between two control actions.

        Args:
            next_control_action: float representing the new control action
            prev_control_action: float representing the previous control action
            t_param: Parameter representing time

        Return:
            ~: Coefficient function
        """

        # TODO: add more documentation, espcially with the t_param - t_param.Get()
        # TODO: rename delta_... to make the equation more readable

        # Define helper variable to make equation easier to read
        delta_control_action = prev_control_action - next_control_action

        # Shift exponential so that t=0 is defined as the present
        t_zeroed = t_param - t_param.Get()

        return CoefficientFunction(next_control_action + delta_control_action * exp(- t_zeroed / self.tau))

    def _evaluate_control_variables(self, soln: GridFunction) -> List[Tuple[float]]:
        """
        Function to measure the values of the control variables.

        Args:
            soln: GridFunction containing the results of the most recent time solve

        Return:
            ~: A list containing the values of the control variables.
               Both scale and vector values will return as a tuple.
        """
        measurements: List[Tuple[float]] = []

        for i in range(len(self.vars_control)):
            var_name = self.vars_control[i][0]
            coords = self.vars_control[i][1]

            # Create the point at which to measure the value at
            if len(coords) == 1:
                point = self.model.mesh(coords[0])
            elif len(coords) == 2:
                point = self.model.mesh(coords[0], coords[1])
            elif len(coords) == 3:
                point = self.model.mesh(coords[0], coords[1], coords[2])
            else:
                raise ValueError("Only 1-3D meshes are supported")

            # Get the index into the gfu corresponding to the variable we want to measure
            var_index = self.model.model_components[var_name]

            # Evaluate the control variable at the point we want
            if var_index is None:
                measurement = soln(point)
            else:
                measurement = soln.components[var_index](point)

            # Convert float values to a tuple of length one so that it's consistent with vector values
            if type(measurement) is float:
                measurement = tuple([measurement])

            # Append solution
            measurements.append(measurement)

        return measurements

    def _update_time_of_next_action(self) -> None:
        """
        Function to update the time at which the next control action occurs.
        This function must be called when

        """
        self.t_next_action += self.dt_control

    @abstractmethod
    def calculate_control_action(self, soln: GridFunction, rk_scheme: bool)\
            -> Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]]:
        """
        Function which calculates what the new value for the manipulated variable should be.

        Args:
            soln: GridFunction containing the results of the most recent time solve
            rk_scheme: bool indicating if the time scheme being used is an RK scheme, and thus requiring the
                recalculation of control actions for all values in t_params
        Return:
            bc_dict_patch: Dictionary containing new values for the BCs representing the manipulated variables
        """
