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

import numpy as np
from numpy import ndarray
import scipy.ndimage as spimg
import scipy.special as spec
from typing import List, Tuple, Union, Dict, Optional
from . import mesh_helpers
from ..helpers.misc import can_import_module

if can_import_module('edt'):
    missing_edt = False
    import edt
else:
    missing_edt = True


def get_binary_2d(boundary_lst: List, N: List[int], scale: List[float], offset: List[float]) -> ndarray:
    """
    Generate a binary representation of a 2D complex geometry on a numpy array.

    This is done by setting the array elements corresponding to points inside the geometry to 1 and all other array
    elements to 0.

    Args:
        boundary_lst: List of coordinates of the geometry's boundary vertices in counterclockwise order.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).

    Returns:
        Array containing a binary representation of the complex geometry.
    """

    shape = (int(N[0] + 1), int(N[1] + 1))
    binary = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = i * scale[0] / N[0] - offset[0]
            y = j * scale[1] / N[1] - offset[1]
            binary[i, j] = mesh_helpers.ray_trace_2d(x, y, boundary_lst)

    binary *= np.ones(shape)

    return binary


def get_binary_3d(face_lst: ndarray, N: List[int], scale: List[float], offset: List[float], mnum: float = 1,
                  close: bool = False) -> ndarray:
    """
    Generate a binary representation of a 3D complex geometry on a numpy array.

    This is done by setting the array elements corresponding to points inside the geometry to 1 and all other array
    elements to 0.

    Args:
        face_lst: List of the vertices and outwards facing normals of the complex geometry's faces.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] cube -> scale=[4,4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] cube -> offset=[2,2,2]).
        mnum: Magic number that increases the distance an array element can be from a vertex while still belonging to
            that vertex. Increase for higher order interpolants if the generated border has gaps.
        close: If True, wraps the binary hole filling in a binary closing. Use if the generated border has gaps.

    Returns:
        Array containing a binary representation of the complex geometry.
    """

    shape = (int(N[0] + 1), int(N[1] + 1), int(N[2] + 1))
    binary = np.zeros(shape)
    for i in range(len(face_lst) - 1):
        n = face_lst[i, :3]
        v1 = face_lst[i, 3:6]
        D = np.dot(n, v1)

        # !!! MAGIC NUMBER !!!
        delta = max(np.array(scale) / np.array(N)) / 2

        x_min = np.min(face_lst[i, np.array([3, 6, 9])])
        x_max = np.max(face_lst[i, np.array([3, 6, 9])])
        y_min = np.min(face_lst[i, np.array([4, 7, 10])])
        y_max = np.max(face_lst[i, np.array([4, 7, 10])])
        z_min = np.min(face_lst[i, np.array([5, 8, 11])])
        z_max = np.max(face_lst[i, np.array([5, 8, 11])])

        x_range = np.linspace(x_min, x_max, int(np.ceil((x_max - x_min) / delta)))
        y_range = np.linspace(y_min, y_max, int(np.ceil((y_max - y_min) / delta)))
        z_range = np.linspace(z_min, z_max, int(np.ceil((z_max - z_min) / delta)))

        if len(x_range) == 0:
            x_range = np.array([x_min])
        if len(y_range) == 0:
            y_range = np.array([y_min])
        if len(z_range) == 0:
            z_range = np.array([z_min])

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if (abs(np.dot(np.array([x, y, z]), n) - D) <= mnum * delta):
                        i = ((x + offset[0]) * N[0] / scale[0]).astype(np.int64)
                        j = ((y + offset[1]) * N[1] / scale[1]).astype(np.int64)
                        k = ((z + offset[2]) * N[2] / scale[2]).astype(np.int64)
                        binary[i, j, k] = 1.0
                    else:
                        pass

    if close:
        binary = spimg.binary_dilation(binary, iterations=1)
        binary = spimg.morphology.binary_fill_holes(binary)
        binary = spimg.binary_erosion(binary, iterations=1)
    else:
        binary = spimg.morphology.binary_fill_holes(binary)

    return binary


def get_phi(binary: ndarray, lmbda: float, N: List[int], scale: List[float], offset: List[float], dim: int = 2) \
        -> ndarray:
    """
    Generate a phase field from a binary representation of a complex geometry.

    The phase field diffuses from 1 inside of the complex geometry to 0 outside of the complex geometry.

    Args:
        binary: Array containing binary representation of complex geometry.
        lmbda: Measure of the diffuseness of the phase field boundary.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        dim: Dimension of the domain (must be 2 or 3).

    Returns:
        Array containing the phase field.
    """
    if missing_edt:
        raise ImportError('edt package not installed, please run `pip install edt`.')

    if dim == 2:
        kernel = np.ones((3, 3))
    elif dim == 3:
        kernel = np.ones((3, 3, 3))
    else:
        raise ValueError('Only works with 2D or 3D meshes.')

    # Use the difference between binary and an eroded binary to get the border
    # of binary, then get the distance transform relative to that border. The
    # distance transform is a Euclidean distance transform that takes into
    # account the entire array, not just a local neighbourhood.
    erosion = spimg.morphology.binary_erosion(binary, kernel, 1).astype(np.float32)
    border = 1.0 - (binary - erosion)
    border = border.astype(np.float32)

    dt = edt.edt(border)
    dt *= min(scale) / min(N)

    # The phase field should run from -1 to 1, take the value 0 at binary's 
    # border, and follow the error function's distribution away from binary's 
    # border (either towards -1 or 1).
    phi_in = spec.erf(dt / lmbda) * binary
    phi_out = spec.erf(dt / lmbda) * (binary - 1.0)
    phi = phi_in + phi_out

    # Modify the phase field to run from 0 to 1.
    phi = (phi + 1.0) / 2.0

    return phi


def nonconformal_subdomain_2d(boundary_lst: List, vertices: List, N: List[int], scale: List[float], offset: List[float],
                              lmbda_overlap: Union[float, bool] = False, centroid: Optional[Tuple[float, float]] = None) \
        -> ndarray:
    """
    Function to generate a 2D BC mask.

    Generate a numpy array that can be used to mask a NGSolve function based on partitioning a mesh's interior domain
    into sections that will have different boundary conditions. Only works in 2D.

    Args:
        boundary_lst: List of coordinates of an interior domain's boundary vertices in counterclockwise order.
        vertices: The coordinates of the two vertices that denote the section of the interior boundary.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        lmbda_overlap: Measure of the diffuseness of the boundary between sections (sharp boundary if False).
        centroid: Coordinates of the point to use as the centroid of the split (centroid of domain if None).

    Returns:
        Mask of domain.
    """

    shape = (int(N[0] + 1), int(N[1] + 1))

    if boundary_lst[0] != boundary_lst[-1]:
        boundary_lst.append(boundary_lst[0])

    # Averaging the vertex coordinates is a naive method of finding the
    # centroid. However, given that the polygon boundary came from a .stl
    # file it can probably be assumed that the boundary points are
    # reasonably evenly spaced and the centroid calculation will not be
    # overly biased.
    if centroid is None:
        cx, cy = np.mean(np.array(boundary_lst), axis=0)
        cx = int(round((cx + offset[0]) * N[0] / scale[0]))
        cy = int(round((cy + offset[1]) * N[1] / scale[1]))
    else:
        cx, cy = centroid
        cx = int(round((cx + offset[0]) * N[0] / scale[0]))
        cy = int(round((cy + offset[1]) * N[1] / scale[1]))

    # A region defined by three points (two vertices and the centroid)
    # contains all points which fall between the two lines defined by each
    # point and the centroid. Since the vertices are ordered
    # counterclockwise, if a third line is defined between the centroid and
    # a point of interest that point only falls between the original two
    # lines if the angle between its line and the line of the first vertex
    # is smaller than the angle between the original two lines.
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x1 = int(round((x1 + offset[0]) * N[0] / scale[0]))
    y1 = int(round((y1 + offset[1]) * N[1] / scale[1]))
    x2 = int(round((x2 + offset[0]) * N[0] / scale[0]))
    y2 = int(round((y2 + offset[1]) * N[1] / scale[1]))

    angle12 = mesh_helpers.angle_between([x1, y1], [cx, cy], [x2, y2])
    mask = np.zeros(shape)
    for j in range(shape[0]):
        for k in range(shape[1]):
            angle1p = mesh_helpers.angle_between([x1, y1], [cx, cy], [j, k])
            if angle1p < angle12:
                mask[j, k] = 1.0

    if not lmbda_overlap:
        pass
    else:
        # The different boundary sections diffuse into each other. Each
        # boundary section is weighted 0.5 at the border between the two
        # sections and diffuses following the error function's
        # distribution.
        dt_in = spimg.distance_transform_bf(mask, 'chessboard', 1).astype(np.float64)
        dt_in /= lmbda_overlap
        dt_in[np.where(dt_in != 0.0)] += (0.5 - 1.0 / lmbda_overlap)
        mask_in = spec.erf(dt_in)

        dt_out = spimg.distance_transform_bf((1.0 - mask), 'chessboard', 1).astype(np.float64)
        dt_out /= lmbda_overlap
        dt_out[np.where(dt_out != 0.0)] += (0.5 - 1.0 / lmbda_overlap)
        mask_out = 1.0 - spec.erf(dt_out)
        mask_out[np.where(mask_out == 1.0)] = 0.0

        mask = mask_in + mask_out

    return mask


def split_nonconformal_subdomains_2d(boundary_lst: List, vertices: Dict, N: List[int], scale: List[float],
                                     offset: List[float], lmbda_overlap: Union[float, bool] = False,
                                     remainder: bool = False, centroid: Dict = {}) -> Dict[str, ndarray]:
    """
    Function to generate all BC masks for a 2D diffuse interface simulation.

    Generate a dictionary of numpy arrays that can be used to mask a NGSolve function based on partitioning a mesh's
    interior domain into sections that will have different boundary conditions. Only works in 2D.

    Args:
        boundary_lst: List of coordinates of an interior domain's boundary vertices in counterclockwise order.
        vertices: Dictionary of coordinates of the vertices that denote the different sections of the interior boundary.
            Vertices must be ordered counterclockwise (unit square with different boundary conditions on each side ->
            vertices={'bottom': [(0,0), (1,0)], 'right': [(1,0), (1,1)], 'top': [(1,1), (0,1)],
            'left': [(0,1), (0,0)]}).
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        lmbda_overlap: Measure of the diffuseness of the boundary between sections (sharp boundary if False).
        remainder: If True a final mask is made from the regions left unmasked by all the previous masks (use if a
            particular boundary section has poorly defined end vertices).
        centroid: Dictionary of the coordinates of the points to use as the centroids of the splits. The dictionary's
            keys should correspond to the boundary condition names like those of vertices.

    Returns:
        Dictionary of numpy array masks.
    """

    shape = (int(N[0] + 1), int(N[1] + 1))

    if not centroid:
        centroid = {marker: None for marker in vertices.keys()}

    mask_dict = {}
    for marker, tmp_vertices in vertices.items():
        mask = nonconformal_subdomain_2d(boundary_lst, tmp_vertices, N, scale, offset, lmbda_overlap, centroid[marker])
        mask_dict[marker] = mask

    if remainder:
        mask = np.ones(shape)
        for marker, m in mask_dict:
            mask -= m

        mask *= (mask > 0.0)
        mask_dict['remainder'] = mask

    return mask_dict


def nonconformal_subdomain_3d(face_lst: ndarray, vertices: str, N: List[int], scale: List[float], offset: List[float],
                              lmbda_overlap: Union[float, bool] = False,
                              centroid: Optional[Tuple[float, float, float]] = None) -> ndarray:
    """
    Function to generate a 3D BC mask.

    Generate a numpy array that can be used to mask a NGSolve function based on partitioning a mesh's interior domain
    into sections that will have different boundary conditions. Only works in 3D.

    Args:
        face_lst: List of the vertices and outwards facing normals of the interior domain's faces.
        vertices: The path to the .msh or .stl file defining the boundary region of the domain.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] cube -> scale=[4,4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] cube -> offset=[2,2,2]).
        lmbda_overlap: Measure of the diffuseness of the boundary between sections (sharp boundary if False).
        centroid: Coordinates of the point to use as the centroid of the split (centroid of domain if None).

    Returns:
        Mask of domain.
    """

    shape = (int(N[0] + 1), int(N[1] + 1), int(N[2] + 1))

    # Averaging the vertex coordinates is a naive method of finding the
    # centroid. However, given that the polygon boundary came from a .stl
    # file it can probably be assumed that the boundary points are
    # reasonably evenly spaced and the centroid calculation will not be
    # overly biased.
    if centroid is None:
        full_vertex_lst = [face[i:i + 3] for face in face_lst for i in [0, 3, 6, 9]]
        centroid = np.mean(np.array(full_vertex_lst), axis=0)

    # Get the boundary edges and vertices of the boundary section.
    edge_lst, boundary_lst = mesh_helpers.get_mesh_boundary_3d(vertices)
    edge_lst = np.array(edge_lst)
    boundary_lst = np.array(boundary_lst)

    # Construct a polygon by connecting the boundary edges of the boundary
    # section to the conformal mesh's centroid. Any points within this
    # polygon (extending the edges to the limits of the nonconformal mesh)
    # will be included in this boundary section's mask. All other points
    # will be set to zero (assumed to belong to a different boundary
    # section).
    # Weight the coordinates of the conformal mesh's centroid equally to
    # the entire averaged coordinates of the boundary section to keep from
    # heavily biasing the polygon's centroid towards the boundary section.
    cx_poly = (np.mean(edge_lst[:, [0, 3]]) + centroid[0]) / 2
    cy_poly = (np.mean(edge_lst[:, [1, 4]]) + centroid[1]) / 2
    cz_poly = (np.mean(edge_lst[:, [2, 5]]) + centroid[2]) / 2
    centroid_poly = np.array([cx_poly, cy_poly, cz_poly])

    # A point is only inside the polygon if for every triangular face
    # comprising the polygon the vector from the point to the face's
    # midpoint is in the opposite direction to the face's outwards facing
    # surface normal.
    mask = np.ones(shape)
    for i in range(len(boundary_lst)):
        p1 = boundary_lst[i, 0:3]
        p2 = boundary_lst[i, 3:6]
        midpoint = (p1 + p2 + centroid) / 3.0

        n = np.cross(p1 - centroid, p2 - centroid)
        n *= (-1) ** (np.dot(midpoint - centroid_poly, n) < 0.0)

        for j in range(shape[0]):
            for k in range(shape[1]):
                for m in range(shape[2]):
                    # Only consider points not already in mask.
                    if mask[j, k, m] != 0.0:
                        p = np.array([j, k, m]) * scale / N - offset
                        if np.dot(p - midpoint, n) > 0.0:
                            mask[j, k, m] = 0.0

    if not lmbda_overlap:
        pass
    else:
        # The different boundary sections diffuse into each other. Each
        # boundary section is weighted 0.5 at the border between the two
        # sections and diffuses following the error function's
        # distribution.
        dt_in = spimg.distance_transform_bf(mask, 'chessboard', 1).astype(np.float64)
        dt_in /= lmbda_overlap
        dt_in[np.where(dt_in != 0.0)] += (0.5 - 1.0 / lmbda_overlap)
        mask_in = spec.erf(dt_in)

        dt_out = spimg.distance_transform_bf((1.0 - mask), 'chessboard', 1).astype(np.float64)
        dt_out /= lmbda_overlap
        dt_out[np.where(dt_out != 0.0)] += (0.5 - 1.0 / lmbda_overlap)
        mask_out = 1.0 - spec.erf(dt_out)
        mask_out[np.where(mask_out == 1.0)] = 0.0

        mask = mask_in + mask_out

    return mask


def split_nonconformal_subdomains_3d(face_lst: List, vertices: Dict[str, str], N: List[int], scale: List[float],
                                     offset: List[float], lmbda_overlap: Union[float, bool] = False,
                                     remainder: bool = False, centroid: Dict = {}) -> Dict[str, ndarray]:
    """
    Function to generate all BC masks for a 3D diffuse interface simulation.

    Generate a list of numpy arrays that can be used to mask a NGSolve function
    based on partitioning a mesh's interior domain into sections that will have
    different boundary conditions. Only works in 3D.

    Args:
        face_lst: List of coordinates of an interior domain's boundary vertices in counterclockwise order.
        vertices: Dictionary of paths to the .msh or .stl files that denote the different sections of the interior
            boundary.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        lmbda_overlap: Measure of the diffuseness of the boundary between sections (sharp boundary if False).
        remainder: If True a final mask is made from the regions left unmasked by all the previous masks (use if a
            particular boundary section has poorly defined end vertices).
        centroid: Dictionary of the coordinates of the points to use as the centroids of the splits. The dictionary's
            keys should correspond to the boundary condition names like those of vertices.

    Returns:
        Dictionary of numpy array masks.
    """

    shape = (int(N[0] + 1), int(N[1] + 1), int(N[2] + 1))

    if not centroid:
        centroid = {marker: None for marker in vertices.keys()}

    mask_dict = {}
    for marker, tmp_vertices in vertices.items():
        assert len(tmp_vertices) == 1
        mask = nonconformal_subdomain_3d(face_lst, tmp_vertices[0],
                                         np.array(N), np.array(scale), np.array(offset),
                                         lmbda_overlap, centroid[marker])
        mask_dict[marker] = mask

    if remainder:
        mask = np.ones(shape)
        for marker, m in mask_dict.items():
            mask -= m

        mask *= (mask > 0.0)
        mask_dict['remainder'] = mask

    return mask_dict
