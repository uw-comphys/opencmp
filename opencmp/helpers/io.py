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

from ngsolve import FESpace, GridFunction, Mesh
from typing import Dict, Optional, List
import os
from ..config_functions import ConfigParser
import ngsolve as ngs
import netgen.meshing as ngmsh
from netgen.read_gmsh import ReadGmsh


def create_and_load_gridfunction_from_file(filename: str, fes: FESpace, current_dir: Optional[List[str]] = None) -> GridFunction:
    """
    Function to create a gridfunction and load the contents of a file into it.

    The gridfunction will be constructed from the same finite element space that the file data is assumed to come
    from. The file data must also fit the exact same dimensions and number of components as the given finite element
    space. NOTE: It is assumed that the gridfunction in the file is from the same FES and mesh as the one passed in,
    there is no way for the code to check. SECOND NOTE: If the FES and mesh are not the same, NGSolve will not
    necessarily crash, it will instead silently return garbage.

    Args:
        filename: Path to the file to load.
        fes: The finite element space the file data was created for.
        current_dir: The directory this command is being called from (ex: dim_dir if being called by DIM). Allows for
            file paths to be given relative to the specific config file's directory instead of the main OpenCMP
            directory only.

    Returns:
        A gridfunction containing the values from the file.
    """

    # Check that the file exists.
    if not os.path.isfile(filename):
        if current_dir is not None:
            for cd in current_dir:
                tmp_filename = cd + '/' + filename
                if os.path.isfile(tmp_filename):
                    filename = tmp_filename
                    break
            else:
                # Did not break out of the for loop so did not find a valid file path.
                raise FileNotFoundError('The file \"{}\" does not exist.'.format(filename))
        else:
            raise FileNotFoundError('The file \"{}\" does not exist.'.format(filename))

    # Load gridfunction from file.
    gfu = GridFunction(fes)
    gfu.Load(filename)

    return gfu


def load_mesh(config: ConfigParser) -> Mesh:
    """
    Loads an NGSolve mesh from a .sol file whose file path is specified in the given config file.

    Args:
        config: A ConfigParser object containing the information from the config file.

    Returns:
        The NGSolve mesh pointed to in the config file.
    """

    assert type(config) is ConfigParser

    try:
        mesh_filename = config['MESH']['filename']
    except KeyError:
        raise ValueError('No default available for MESH, filename. Please specify a value in the config file.')

    # Check that the file exists.
    if not os.path.isfile(mesh_filename):
        raise FileNotFoundError('The given mesh file \"{}\" does not exist.'.format(mesh_filename))

    # Mesh can be a Netgen mesh or a GMSH mesh.
    if mesh_filename.endswith('.msh'):
        ngmesh = ReadGmsh(mesh_filename)
        mesh = ngs.Mesh(ngmesh)
    elif mesh_filename.endswith('.vol'):
        ngmesh = ngmsh.Mesh()
        ngmesh.Load(mesh_filename)

        mesh = ngs.Mesh(ngmesh)
    else:
        raise TypeError('Only .vol (Netgen) and .msh (GMSH) meshes can be used.'
                        'Your specified filename was \"{}\"'.format(mesh_filename))

    # Suppressing the warning about using the default value for curved_elements.
    curved_elements = config.get_item(['MESH', 'curved_elements'], bool, quiet=True)
    if curved_elements:
        interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)
        mesh.Curve(interp_ord)

    return mesh


def update_gridfunction_from_files(gfu: GridFunction, file_dict: Dict[Optional[int], str]) -> None:
    """
    Function to take an existing gridfunction and load data into it from one or more files.

    NOTE: It is assumed that the save data in the files is from the same FES and mesh as the existing grid function.
    There is no way to check this in code. SECOND NOTE: If the FES and mesh are not the same, NGSolve will not
    necessarily crash, it will instead silently return garbage.

    Args:
        file_dict: Dict containing the paths to the files
        gfu: The gridfunction to load the values into
    """

    for key, val in file_dict.items():
        # Check that the file exists.
        if not os.path.isfile(val):
            raise FileNotFoundError('The given file does not exist.')

        if key is None:  # A single gridfunction
            # Confirm that file_dict only has one value, otherwise the gfu values will be overwritten multiple times
            assert len(file_dict) == 1

            gfu.Load(val)
        else:
            # The values for the various components of the gridfunction were saved separately.
            gfu.components[key].Load(val)
