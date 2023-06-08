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

import pyngcore as ngcore
from ngsolve import GridFunction

from .output_conversions import sol_to_vtu, sol_to_vtu_direct, sol_to_components
from .error_analysis import convergence_analysis
from ..config_functions import ConfigParser
from ..helpers.error import calc_error
from ..solvers import Solver


def run_post_processing(config_parser: ConfigParser, solver: Solver, sol: GridFunction) -> None:
    """
    This function iterates through each of the built-in post-processing operations, checks if they
    should be run, and then runs them.

    Args:
        config_parser:  The config_parser for the simulation to perform post-processing on
        solver:         The solver object created from the config_parser and which produced sol
        sol:            Gridfunction containing the final solution produced by the model from solver.model
    """
    with ngcore.TaskManager():
        # Calculate error metrics on the solution (e.g. L2 norm or divergence of the velocity field)
        if config_parser.get_item(['ERROR ANALYSIS', 'check_error'], bool):
            calc_error(config_parser, solver.model, sol)

        convergence_analysis(config_parser, solver, sol)

    save_output = config_parser.get_item(['VISUALIZATION', 'save_to_file'], bool, quiet=True)
    save_type = config_parser.get_item(['VISUALIZATION', 'save_type'], str, quiet=True)
    # Run the post-processor to convert the .sol to .vtu
    if save_output and save_type == '.vtu':
        print('Converting saved output to VTU.')
        sol_to_vtu(config_parser, solver)

    # Split the .sol file for the final time-step into individual components to make using it for
    # the initial conditions of other simulations easier
    if config_parser.get_item(['VISUALIZATION', 'split_components'], bool, quiet=True):
        sol_to_components(config_parser,
                          config_parser.get_item(['OTHER', 'run_dir'], str) + '/output/',
                          solver.model)

