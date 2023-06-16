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

import logging
from typing import Optional, cast

from ngsolve import ngsglobals
import pyngcore as ngcore

from .models import get_model_class
from .solvers import get_solver_class
from .config_functions import ConfigParser
from .post_processing import run_post_processing


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

    # configure logging and status output
    logging.basicConfig(filename='opencmp.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # Set parameters for ngsolve
    ngcore.SetNumThreads(num_threads)
    ngsglobals.msg_level = msg_level

    # Run the model.
    with ngcore.TaskManager():
        dim_used = config_parser.get_item(['DIM', 'diffuse_interface_method'], bool, quiet=True)

        model_class = get_model_class(model_name, dim_used)
        solver_class = get_solver_class(config_parser)
        solver = solver_class(model_class, config_parser)
        sol = solver.solve()

    run_post_processing(config_parser, solver, sol)

    return
