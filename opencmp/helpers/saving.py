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

import re
from typing import Union
from ngsolve import GridFunction, CoefficientFunction, VTKOutput
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
        self.save_dir_sol = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output/' + model.name() + '_sol/'
        self.save_dir_vtu = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output/' + model.name() + '_vtu/'

        # Specifically for diffuse interface rigid body motion.
        self.save_dir_phi = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output_phi/'
        self.save_dir_phi_sol = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output_phi/' + model.name() + '_sol/'
        self.save_dir_phi_vtu = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output_phi/' + model.name() + '_vtu/'

        self.base_filename_sol = self.save_dir_sol + model.name() + '_'
        self.base_filename_phi_sol = self.save_dir_phi_sol + 'phi' + '_'
        self.base_subdivision = model.config.get_item(['VISUALIZATION', 'subdivision'], int, quiet)
        self.save_vtu_each_timestep = (
            base_type == '.vtu'
            and model.config.get_item(
                ['VISUALIZATION', 'save_vtu_each_timestep'], bool, quiet)
        )
        self.model_name = model.name()
        self.mesh = model.mesh
        self.save_names = model.save_names
        self.pvd_entries = {False: {}, True: {}}

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

        if self.save_vtu_each_timestep:
            self._save_vtu(gfu, timestep, DIM)

    def _save_vtu(self, gfu: Union[GridFunction, CoefficientFunction], timestep: float, DIM: bool) -> None:
        """Write the just-saved checkpoint to VTU and refresh its PVD collection."""
        if DIM:
            vtu_dir = self.save_dir_phi_vtu
            basename = 'phi_' + str(timestep)
            names = ['phi']
            pvd_filename = self.save_dir_phi + 'phi_transient.pvd'
            relative_filename = self.model_name + '_vtu/' + basename + '.vtu'
        else:
            vtu_dir = self.save_dir_vtu
            basename = self.model_name + '_' + str(timestep)
            names = self.save_names
            pvd_filename = self.save_dir + self.model_name + '_transient.pvd'
            relative_filename = self.model_name + '_vtu/' + basename + '.vtu'

        # On the first write of a run, keep any entries already in the .pvd so a
        # resumed run doesn't truncate the collection to just the new timesteps.
        if not self.pvd_entries[DIM] and Path(pvd_filename).is_file():
            self.pvd_entries[DIM] = {float(t): f for t, f in
                                     re.findall(r'timestep="([^"]+)".*?file="([^"]+)"',
                                                Path(pvd_filename).read_text())}

        coefs = list(gfu.components) if isinstance(gfu, GridFunction) and len(gfu.components) > 0 else [gfu]
        VTKOutput(ma=self.mesh, coefs=coefs, names=names,
                  filename=vtu_dir + basename, subdivision=self.base_subdivision).Do()

        self.pvd_entries[DIM][float(timestep)] = relative_filename
        with open(pvd_filename, 'w') as pvd:
            pvd.write('<?xml version="1.0"?>\n'
                      '<VTKFile type="Collection" version="0.1"\n'
                      'byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n'
                      '<Collection>\n')
            for time, path in sorted(self.pvd_entries[DIM].items()):
                pvd.write('<DataSet timestep="%e" group="" part="0" file="%s"/>\n'
                          % (time, path))
            pvd.write('</Collection>\n</VTKFile>')
