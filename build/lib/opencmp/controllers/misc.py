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
"""
Module containing helper functions related to models
"""
from typing import List

from . import controllers_dict
from ..models import Model
from .base_controller import Controller
from ngsolve import Parameter


def get_controller(controller_type: str, t_params: List[Parameter], model: Model, config_rel_path: str) -> Controller:
    """
    Function to get the controller to use.

    Args:
        controller_type: The name of the controller to initialize
        t_params: An ngsolve Parameter representing the current time
        model: The model to hook the controller up to
        config_rel_path: The filename, and relative path, for the config file for this controller

    Returns:
        ~: An initialized controller of the type specified.
    """

    # Find the controller by its name
    controller_class = controllers_dict[controller_type]

    # Check that whatever we found is actually a model
    if not issubclass(controller_class, Controller):
        # Raise an error if it isn't
        raise AttributeError("Provided class is not a controller")

    # Initialize and return the controller
    return controller_class(t_params, model, config_rel_path, model.run_dir)

