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

from . import *
from .adaptive_transient_solvers import *
from typing import Type
from ..config_functions import ConfigParser

"""
Module containing helper functions related to solvers.
"""

def get_solver_class(config: ConfigParser) -> Type[Solver]:
    """
    Function to get the solver to use.

    Args:
        config: The config file from which to get information for which solver to use.

    Returns:
        The solver to use.
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
            if scheme in ['explicit euler', 'implicit euler', 'crank nicolson', 'euler IMEX', 'CNLF', 'SBDF']:
                solver_class = TransientMultiStepSolver
            elif scheme in ['RK 222', 'RK 232']:
                solver_class = TransientRKSolver
            else:
                raise TypeError('Have not implemented {} time integration yet.'.format(scheme))
    else:
        solver_class = StationarySolver

    return solver_class
