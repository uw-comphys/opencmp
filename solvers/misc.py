"""
Module containing helper functions related so solvers.
"""

from . import *
from .adaptive_solvers import *
from typing import Type
from config_functions import ConfigParser


def get_solver_class(config: ConfigParser) -> Type[Solver]:
    """
    Function to get the solver to use.

    Args:
        config: The config file from which to get information for which solver to use.

    Returns:
        ~: The solver to use.
    """
    solver_class: Type[Solver]

    transient = config.get_item(['TRANSIENT', 'transient'], bool, quiet=True)
    scheme = config.get_item(['TRANSIENT', 'scheme'], str, quiet=True)
    adaptive = 'adaptive' in scheme

    if transient:
        if adaptive:
            if scheme == 'adaptive two step':
                solver_class = AdaptiveTwoStep
            elif scheme == 'adaptive three step':
                solver_class = AdaptiveThreeStep
            elif scheme == 'adaptive IMEX':
                solver_class = AdaptiveIMEX
            else:
                raise TypeError('Have not implemented {} time integration yet.'.format(scheme))
        else:
            solver_class = TransientSolver
    else:
        solver_class = StationarySolver

    return solver_class
