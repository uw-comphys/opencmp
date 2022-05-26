# Entry points (console commands) for OpenCMP

import os, sys, platform, time
from typing import Dict, Optional, cast
from .models import get_model_class
from .solvers import get_solver_class
from .helpers.error_analysis import h_convergence, p_convergence
from .helpers.error import calc_error
from .config_functions import ConfigParser
import pyngcore as ngcore
from ngsolve import ngsglobals
from .helpers.post_processing import sol_to_vtu, PhaseFieldModelMimic
from .run import run

def run_opencmp():
    """
    Main function that runs OpenCMP.

    Args (from command line):
        config_file_path: Filename of the config file to load. Required parameter.
        config_parser: Optionally provide the ConfigParser if running tests. Optional parameter.
    """
    
    # Arguments for command line use

    config_parser = None

    if len(sys.argv) == 1: # if user did not provide any configuration path (which is required)
        print("ERROR: Provide configuration file path.")
        exit(0)

    elif len(sys.argv) == 2: # if user did provide a configuration path
        config_file_path = sys.argv[1]
        config_parser = ConfigParser(config_file_path)

    elif len(sys.argv) == 3: # if the user provided both config path and optional config_parser argument
        config_file_path = sys.argv[1]
        config_parser = sys.argv[2]
        config_parser = cast(ConfigParser, config_parser)    
    
    else: # if the user provides more than 2 arguments. Print error messages and quit.
        print('ERROR: More than two arguments were provided.')
        print('\tOpenCMP supports up to two (2) arguments. The first argument is a required configuration file path.')
        print('\tThe second argument is the optional ConfigParser if running tests.')
        print('\tPlease re-try using only 1 or 2 arguments as described above. Thank you.')
        exit(0)

    # call the function in run.py
    run(config_file_path, config_parser)

    return

# Entry point 1: install all optional dependencies for OpenCMP: edit, tabulate, pytest
def install_optional_dependencies():
    
    print("Now installing OpenCMP optional dependencies: edt, tabulate, pytest.")
    os.system("pip3 install edt tabulate pytest")


# Entry point 2: short form command to run the pytests... instead of doing python -m pytest pytests/
def pytest_tests():
    
    print("Now will run the pytests...")

    user_os = str(platform.system())

    if user_os == 'Windows':
        os.system("py -m pytest pytests//")
    
    else: # macOS or linux (and WSL)
        os.system("python3 -m pytest pytests//")

# Entry point 3: install pytest...
def install_pytest():

    os.system("pip3 install pytest")
