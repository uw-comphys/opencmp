from typing import List

import ngsolve as ngs
import numpy as np
from ngsolve import CoefficientFunction, FESpace, GridFunction, Mesh

from config_functions import ConfigParser
from models import Model


def nondimensionalize_mesh_file(filename: str, char_length: List[float]) -> None:
    """
    Function to nondimensionalize a mesh file.

    Given the path to a .vol or .msh file, a new file is created containing the exact same mesh but now
    nondimensionlized by the given characteristic length scale.

    Args:
        filename: The path to the mesh file.
        char_length: The characteristic length scale. Can be different for each dimension
                     (ex: char_length = [x_scale, y_scale, z_scale]) or if a single value is given it is used for every
                     dimension (ex: char_length = [scale] -> char_length = [scale, scale, scale]).
    """
    # TODO
    return

def nondimensionlize_loaded_mesh(mesh: ngs.Mesh, char_length: List[float]) -> ngs.Mesh:
    """
    Function to nondimensionalize an NGSolve mesh.

    Given an NGSolve mesh, the mesh nodes are modified to nondimensionlize it by the given characteristic length scale.

    Args:
        mesh: The mesh to nondimensionalize.
        char_length: The characteristic length scale. Can be different for each dimension
                     (ex: char_length = [x_scale, y_scale, z_scale]) or if a single value is given it is used for every
                     dimension (ex: char_length = [scale] -> char_length = [scale, scale, scale]).

    Returns:
        mesh: The original mesh now nondimensionalized.
    """

    for p in mesh.ngmesh.Points():
        px, py = p[0], p[1]
    p[0] = pxnp.cos(theta) - pynp.sin(theta)
    p[1] = pxnp.sin(theta) + pynp.cos(theta)
    if p[0] > maxX:
        maxX = p[0]
    if p[0] < minX:
        minX = p[0]
    if p[1] > maxY:
        maxY = p[1]
    if p[1] < minY:
        minY = p[1]