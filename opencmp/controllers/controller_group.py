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

from .base_controller import Controller
from .misc import get_controller
from ..config_functions import ConfigParser
from ..helpers import merge_bc_dict
from ..models import Model
from ngsolve import CoefficientFunction, GridFunction, Parameter
from typing import Dict, List, Optional


class ControllerGroup:
    """
    Class which holds multiple controllers
    """

    def __init__(self, t_params: List[Parameter], model: Model, main_config: ConfigParser) -> None:
        """
        Initializer.

        Args:
            t_params: List of time parameters representing the current and past times
            model: The model that the control is being applied to
            main_config: The ConfigParser object for the main config file
        """
        # List of all controllers
        self.controllers: List[Controller] = []

        # Get controller types and corresponding config files
        controller_types = main_config.get_list(['CONTROLLER', 'type'], str)
        controller_config_filenames = main_config.get_list(['CONTROLLER', 'config'], str)

        # The directory in which all control config files for this simulation are stored
        control_dir = main_config.get_item(['OTHER', 'run_dir'], str) + '/control_dir/'

        # Check that enough controllers/config files were provided
        assert len(controller_types) == len(controller_config_filenames)

        # Iterate over each controller and create it
        for i in range(len(controller_types)):
            controller_type = controller_types[i]
            controller_rel_path = control_dir + controller_config_filenames[i]

            self.controllers.append(get_controller(controller_type, t_params, model, controller_rel_path))

    def calculate_control_all_actions(self, soln: GridFunction, rk_scheme: bool = False) \
            -> Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]]:
        """
        Calculate all of the control actions taken at this timestep.

        NOTE: If two or more models try to change the same boundary condition, the final value will be that of the last
        controller that tried to change it.

        DO NOT rely on this behaviour. If you use more than one controller, make sure that they are

        Args:
            soln: GridFunction containing the results of the most recent time solve
            rk_scheme: bool indicating if the time scheme being used is an RK scheme, and thus requiring the
                recalculation of control actions for all values in t_params

        Return:
            bc_dict_patch: Dictionary containing new values for the BCs representing the manipulated variables
        """
        bc_dict_patch: Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]] = {}

        for controller in self.controllers:
            bc_dict_patch = merge_bc_dict(bc_dict_patch, controller.calculate_control_action(soln, rk_scheme))

        return bc_dict_patch

    def get_time_for_next_two_control_actions(self) -> List[float]:
        """
        Function to obtain the times for the next two control actions.
        Used by the solvers to adjust timesteps to ensure that the model is solved at the times required for control

        Returns:
            ~: The time at which the next control action will occur
        """
        times: List[float] = []

        for controller in self.controllers:
            # Get the next time for the control action
            t_1st_next = controller.t_next_action
            # Calculate the time for the control action after that
            t_2nd_next = t_1st_next + controller.dt_control

            # Add both
            times.append(t_1st_next)
            times.append(t_2nd_next)

        return times
