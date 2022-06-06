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

import os, sys, platform
from .run import run

def run_opencmp():
    """
    Main function that runs OpenCMP.

    Args (from command line):
        config_file_path: Filename of the config file to load. Required parameter.

    """
    
    # Arguments for command line use

    if len(sys.argv) == 1: # if user did not provide any configuration path (which is required)
        print("ERROR: Provide configuration file path.")
        exit(0)

    elif len(sys.argv) == 2: # if user did provide a configuration path
        config_file_path = sys.argv[1]

    else: # if the user provides more than 1 argument (in addition to opencmp). Print error messages and quit.
        print('ERROR: More than one argument was provided.')
        print('** OpenCMP supports up to one (1) argument. The argument is a required configuration file path.')
        print('Please re-try with "opencmp config" where "config" is the name of the configuration file in the current directory. **')
        exit(0)

    # call the function in run.py
    run(config_file_path)

    return


def pytest_tests():
    
    print("Now will run the pytests...")

    user_os = str(platform.system())

    if user_os == 'Windows':
        os.system("py -m pytest pytests//")
    
    else: # macOS or linux (and WSL)
        os.system("python3 -m pytest pytests//")
