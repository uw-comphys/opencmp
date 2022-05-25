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

from typing import Union
from ngsolve import GridFunction, CoefficientFunction
from ..models import Model
from pathlib import Path


class SolutionFileSaver:
    """
    Class to handle the saving of GridFunctions and CoefficientFunctions to file
    """

    def __init__(self, model: Model, quiet: bool = False) -> None:
        """
        Initializer

        Args:
            model: The model being solved from which to get necessary information.
            quiet: If True suppresses the warning about the default value being used for a parameter.
        """

        # Check that only valid output types were passed
        base_type = model.config.get_item(['VISUALIZATION', 'save_type'], str, quiet)
        if base_type not in ['.sol', '.vtu']:
            print('Can\'t output to file type {}.'.format(base_type))

        self.save_dir = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output/'
        self.save_dir_sol = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output/sol/'
        self.save_dir_vtu = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output/vtu/'

        # Specifically for diffuse interface rigid body motion.
        self.save_dir_phi = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output_phi/'
        self.save_dir_phi_sol = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output_phi/sol/'
        self.save_dir_phi_vtu = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output_phi/vtu/'

        self.base_filename_sol = self.save_dir_sol + model.name + '_'
        self.base_filename_phi_sol = self.save_dir_phi_sol + 'phi' + '_'
        self.base_subdivision = model.config.get_item(['VISUALIZATION', 'subdivision'], int, quiet)

        # Create the save dir if it doesn't exist
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_dir_sol).mkdir(parents=True, exist_ok=True)
        if base_type == '.vtu':
            Path(self.save_dir_vtu).mkdir(parents=True, exist_ok=True)

        # Add save dirs for the phase field if the diffuse interface method is being used for rigid body motion and
        # they don't exist.
        if model.DIM:
            Path(self.save_dir_phi).mkdir(parents=True, exist_ok=True)
            Path(self.save_dir_phi_sol).mkdir(parents=True, exist_ok=True)
            if base_type == '.vtu':
                Path(self.save_dir_phi_vtu).mkdir(parents=True, exist_ok=True)

        # NOTE: -1 is the value used whenever an int default is needed.
        if self.base_subdivision == -1:
            self.base_subdivision = model.interp_ord

    def save(self, gfu: Union[GridFunction, CoefficientFunction], timestep: float, DIM=False) -> None:
        """
        Function to save the provided GridFunction or CoefficientFunction to file.

        Args:
            gfu: GridFunction or CoefficientFunction to save
            timestep: The current time step, used for naming the file
            DIM: If True, a phase field is being saved so should be saved to the phi_sol directory.
        """

        # Assemble filename
        if not DIM:
            # Solution gridfunction so save to the normal sol directory.
            filename = self.base_filename_sol + str(timestep) + '.sol'
        else:
            # Phase field gridfunction so save to phi_sol directory.
            filename = self.base_filename_phi_sol + str(timestep) + '.sol'

        # Save to file
        gfu.Save(filename)
