from ngsolve import FESpace, GridFunction, Mesh
from typing import Dict, Optional
import os
from config_functions import ConfigParser
import ngsolve as ngs
import netgen.meshing as ngmsh
from netgen.read_gmsh import ReadGmsh


def load_file_into_gridfunction(gfu: GridFunction, file_dict: Dict[Optional[int], str]) -> None:
    """
    Function to load data from a file(s) into a gridfunction. It is assumed that the saved data comes from the same
    finite element space and the same mesh as the gridfunction being returned.

    Args:
        file_dict: Dict containing the paths to the files
        gfu: The gridfunction to load the values into

    Returns:
        gfu: A gridfunction containing the values from the file
    """

    for key, val in file_dict.items():
        if key is None:
            # One single gridfunction.
            assert len(file_dict) == 1  # Confirm that file_dict only has one value, otherwise the gfu values will be overwritten multiple times
            gfu.Load(val)

        else:
            # The values for the various components of the gridfunction were saved separately.
            gfu.components[key].Load(val)


def load_gridfunction_from_file(filename: str, fes: FESpace) -> GridFunction:
    """
    Function to load a gridfunction from a file. The gridfunction will be constructed from the same finite element
    space that the file data is assumed to come from. The file data must also fit the exact same dimensions and number
    of components as the given finite element space.

    Args:
        filename: File to load.
        fes: The finite element space the file data was created for.

    Returns:
        gfu: A gridfunction containing the values from the file
    """

    # Check that the file exists.
    if not os.path.isfile(filename):
        raise FileNotFoundError('The given file does not exist.')

    # Load gridfunction from file.
    gfu = GridFunction(fes)
    gfu.Load(filename)

    return gfu


def load_mesh(config: ConfigParser) -> Mesh:
    """
    Loads an NGSolve mesh from a .sol file whose file path is specified in the
    given config file.

    Args:
        config: A ConfigParser object containing the information from the config file.

    Returns:
        mesh: The NGSolve mesh pointed to in the config file.
    """

    try:
        mesh_filename = config['MESH']['filename']
    except:
        raise ValueError('No default available for MESH, filename. Please specify a value in the config file.')

    # Check that the file exists.
    if not os.path.isfile(mesh_filename):
        raise FileNotFoundError('The given mesh file does not exist.')

    # Mesh can be a Netgen mesh or a GMSH mesh.
    if mesh_filename.endswith('.msh'):
        ngmesh = ReadGmsh(mesh_filename)
        mesh = ngs.Mesh(ngmesh)
    elif mesh_filename.endswith('.vol'):
        ngmesh = ngmsh.Mesh()
        ngmesh.Load(mesh_filename)

        mesh = ngs.Mesh(ngmesh)

        # Suppressing the warning about using the default value for curved_elements.
        curved_elements = config.get_item(['MESH', 'curved_elements'], bool, quiet=True)
        if curved_elements:
            interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)
            mesh.Curve(interp_ord)
    else:
        raise TypeError('Only .vol (Netgen) and .msh (GMSH) meshes can be used.')

    return mesh
