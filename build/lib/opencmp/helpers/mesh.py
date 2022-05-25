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

from typing import List

from ngsolve import Mesh


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
    raise NotImplemented


def nondimensionlize_loaded_mesh(mesh: Mesh, char_length: List[float]) -> Mesh:
    """
    Function to nondimensionalize an NGSolve mesh.

    Given an NGSolve mesh, the mesh nodes are modified to nondimensionlize it by the given characteristic length scale.

    Args:
        mesh: The mesh to nondimensionalize.
        char_length: The characteristic length scale. Can be different for each dimension
            (ex: char_length = [x_scale, y_scale, z_scale]) or if a single value is given it is used for every
            dimension (ex: char_length = [scale] -> char_length = [scale, scale, scale]).

    Returns:
        The original mesh now nondimensionalized.
    """

    # TODO
    raise NotImplemented

    # for p in mesh.ngmesh.Points():
    #     px, py = p[0], p[1]
    #     # p[0] = px*np.cos(theta) - py*np.sin(theta)
    #     # p[1] = px*np.sin(theta) + py*np.cos(theta)
    #
    #     if p[0] > maxX:
    #         maxX = p[0]
    #     if p[0] < minX:
    #         minX = p[0]
    #     if p[1] > maxY:
    #         maxY = p[1]
    #     if p[1] < minY:
    #         minY = p[1]
