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

# Superclass
from .base_model import Model

# Implemented models
from .ins                   import INS
from .ins_dim               import INSDIM
from .poisson               import Poisson
from .poisson_dim           import PoissonDIM
from .stokes                import Stokes
from .stokes_dim            import StokesDIM
from .multi_component_ins   import MultiComponentINS

models_dict = {"INS": INS,
               "INS-DIM": INSDIM,
               "Poisson": Poisson,
               "Poisson-DIM": PoissonDIM,
               "Stokes": Stokes,
               "Stokes-DIM": StokesDIM,
               "MultiComponentINS": MultiComponentINS}

# Helper functions
from .misc import get_model_class
