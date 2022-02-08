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

from typing import Dict, Optional, cast

from .models import get_model_class
from .solvers import get_solver_class
from .helpers.error_analysis import h_convergence, p_convergence
from .helpers.error import calc_error
from .config_functions import ConfigParser
import pyngcore as ngcore
from ngsolve import ngsglobals
from .helpers.post_processing import sol_to_vtu, PhaseFieldModelMimic


def run(config_file_path: str, config_parser: Optional[ConfigParser] = None) -> None:
    """
    Main function that runs OpenCMP.

    Args:
        config_file_path: Filename of the config file to load.
        config_parser: Optionally provide the ConfigParser if running tests.
    """
    # Load the config_parser file.
    if config_parser is None:
        config_parser = ConfigParser(config_file_path)
    else:
        assert config_parser is not None
        config_parser = cast(ConfigParser, config_parser)

    # Load run parameters from the config_parser file.
    num_threads = config_parser.get_item(['OTHER', 'num_threads'], int)
    msg_level = config_parser.get_item(['OTHER', 'messaging_level'], int, quiet=True)
    model_name = config_parser.get_item(['OTHER', 'model'], str)

    # Load error analysis parameters from the config_parser file.
    check_error = config_parser.get_item(['ERROR ANALYSIS', 'check_error'], bool)

    # Set parameters for ngsolve
    ngcore.SetNumThreads(num_threads)
    ngsglobals.msg_level = msg_level

    # Run the model.
    with ngcore.TaskManager():

        model_class = get_model_class(model_name)
        solver_class = get_solver_class(config_parser)
        solver = solver_class(model_class, config_parser)
        sol = solver.solve()

        if check_error:
            calc_error(config_parser, solver.model, sol)

        # Suppressing the warning about using the default value for convergence_test.
        convergence_test: Dict[str, str] = config_parser.get_dict(['ERROR ANALYSIS', 'convergence_test'],
                                                                  None, quiet=True)
        for key, var_lst in convergence_test.items():
            if key == 'h' and var_lst:
                for var in var_lst:
                    h_convergence(config_parser, solver, sol, var)
            elif key == 'p' and var_lst:
                for var in var_lst:
                    p_convergence(config_parser, solver, sol, var)

    save_output = config_parser.get_item(['VISUALIZATION', 'save_to_file'], bool, quiet=True)
    if save_output:
        save_type = config_parser.get_item(['VISUALIZATION', 'save_type'], str, quiet=True)

        # Run the post-processor to convert the .sol to .vtu
        if save_type == '.vtu':
            print('Converting saved output to VTU.')

            # Path where output is stored
            output_dir_path = config_parser.get_item(['OTHER', 'run_dir'], str) + '/output/'

            # Run the conversion
            sol_to_vtu(config_parser, output_dir_path, solver.model)

            # Repeat for the saved phase field .sol files if using the diffuse interface method.
            if solver.model.DIM:
                print('Converting saved phase fields to VTU.')

                # Construct a mimic of the Model class appropriate for the phase field (mainly contains the correct
                # finite element space).
                phi_model = PhaseFieldModelMimic(solver.model)

                # Path where the output is stored
                output_dir_phi_path = config_parser.get_item(['OTHER', 'run_dir'], str) + '/output_phi/'

                # Run the conversion.
                # Note: The normal main simulation ConfigParse can be used since it is only used
                # to get a value for subdivision.
                sol_to_vtu(config_parser, output_dir_phi_path, phi_model)

    return
