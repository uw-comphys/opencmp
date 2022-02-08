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
import netgen.meshing as ngmsh
from netgen.read_gmsh import ReadGmsh
import os
from typing import List, Any, Tuple


def index_sublist(lst: List, val: Any) -> int:
    """
    Find which sublist of a list contains a given value.

    Args:
        lst: List to search through.
        val: Value to check for.

    Returns:
        Index of lst corresponding to the sublist of interest.
    """

    for i in range(len(lst)):
        if val in lst[i]:
            return i

    raise RuntimeError('lst does not contain val.')


def angle_between(p1_tmp: List[float], p2_tmp: List[float], p3_tmp: List[float]) -> float:
    """
    Calculate the angle formed by p1-p2-p3.

    Args:
        p1_tmp: [x,y] or [x,y,z] coordinate.
        p2_tmp: [x,y] or [x,y,z] coordinate.
        p3_tmp: [x,y] or [x,y,z] coordinate.

    Returns:
        Calculated angle in radians rescaled to [0,2*pi].
    """

    p1 = np.array(p1_tmp)
    p2 = np.array(p2_tmp)
    p3 = np.array(p3_tmp)

    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))

    if angle < 0.0:
        angle += 2 * np.pi

    return angle


def move_vertex(vertex: List, vertex_lst: List, vertex_coords: List) \
        -> Tuple[List, List, int]:
    """
    Find vertex pair [A,B] (or [B,A]) containing vertex A and return vertex B of said pair.

    Args:
        vertex: Vertex used to find vertex pair.
        vertex_lst: List of vertex pairs.
        vertex_coords: List of coordinates corresponding to vertex pairs in vertex_lst.

    Returns:
        Tuple[List, List, int]:
            - v_prime: Vertex B of vertex pair [A,B] (or [B,A]).
            - p_prime: [x,y] or [x,y,z] coordinates of vertex B.
            - index: Index of vertex pair [A,B] (or [B,A]) in vertex_lst.
    """

    index = index_sublist(vertex_lst, vertex)
    if vertex_lst[index][0] == vertex:
        v_prime = vertex_lst[index][1]
        p_prime = vertex_coords[index][2:]
    elif vertex_lst[index][1] == vertex:
        v_prime = vertex_lst[index][0]
        p_prime = vertex_coords[index][:2]

    return v_prime, p_prime, index


def reorder_vertices_2d(edge_lst: List, vertex_lst: List, vertex_coords: List) -> List:
    """
    Rearrange a list of mesh boundary vertices to be in order based on the mesh's boundary edge connectivities.

    Args:
        edge_lst: List of mesh boundary edges.
        vertex_lst: List of pairs of mesh boundary vertices corresponding to the mesh's boundary edges.
        vertex_coords: List of coordinates corresponding to vertex pairs in vertex_lst.

    Returns:
        Ordered list of coordinates of mesh boundary vertices (could be clockwise or counterclockwise order).
    """

    hull = []
    current_coords = edge_lst.pop(0)
    current_vertex, next_vertex = vertex_lst.pop(0)
    current_coords, next_coords = vertex_coords.pop(0)
    hull.append(current_coords)
    for i in range(len(edge_lst)):
        index = index_sublist(vertex_lst, next_vertex)
        vertices = vertex_lst.pop(index)
        if vertices[0] == next_vertex:
            current_vertex, next_vertex = vertices
            current_coords, next_coords = vertex_coords.pop(index)
        else:
            next_vertex, current_vertex = vertices
            next_coords, current_coords = vertex_coords.pop(index)
        hull.append(current_coords)
    hull.append(hull[0])

    return hull


def signed_area(points_lst: List) -> float:
    """
    Calculate the signed area of a non-intersecting polygon.

    Args:
        points_lst: Ordered list of coordinates of polygon vertices.

    Returns:
        Signed area of polygon.
    """

    area = 0.0

    for i in range(len(points_lst) - 1):
        x1, y1 = points_lst[i]
        x2, y2 = points_lst[i + 1]
        area += (x1 * y2 - x2 * y1)

    area /= 2.0

    return area


def order_ccw(points_lst: List) -> List:
    """
    Rearrange an ordered list of polygon vertices to be in counterclockwise order.

    Args:
        points_lst: Ordered list of coordinates of polygon vertices (could be in clockwise or counterclockwise order).

    Returns:
        points_lst: List of coordinates of polygon vertices in counterclockwise order.
    """

    ccw = signed_area(points_lst)

    if ccw < 0:
        points_lst = points_lst[::-1]

    return points_lst


def orient_2d(p1: List[float], p2: List[float], p3: List[float], eps: float = 1.0e-10) -> str:
    """
    Check if a set of three 2D points is collinear, ordered counterclockwise, or ordered clockwise.

    Args:
        p1: [x,y] coordinate.
        p2: [x,y] coordinate.
        p3: [x,y] coordinate.
        eps: Tolerance on whether or not the points are collinear.

    Returns:
        Indicates the ordering of the points.
    """

    test = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])

    if abs(test) <= eps:
        return 'collinear'
    elif test > 0.0:
        return 'cw'
    else:
        return 'ccw'


def orient_3d(p: ndarray, v1: ndarray, v2: ndarray, v3: ndarray, eps: float = 1e-6) -> int:
    """
    Determine which side of the plane defined by v1, v2, v3 the point p is on.

    This function assumes a point and plane in 3D space.

    Args:
        p: [x,y,z] coordinate.
        v1: [x,y,z] coordinate of one of the plane's vertices.
        v2: [x,y,z] coordinate of one of the plane's vertices.
        v3: [x,y,z] coordinate of one of the plane's vertices.
        eps: Tolerance for adding small perturbations to the plane's vertices if p is degenerate with an edge or vertex
            of the plane.

    Returns:
        Denotes which side of the plane p is on.
    """

    cross_prod = np.cross(v1 - p, v2 - p)
    dot_prod = np.dot(cross_prod, v3 - p)
    volume = np.sign(dot_prod)

    return volume


def calc_barycentric(p: ndarray, v1: ndarray, v2: ndarray, v3: ndarray) -> bool:
    """
    Use barycentric coordinates to determine if point p, which lies on the plane with normal n, also lies within the
    triangle v1-v2-v3.

    Args:
        p: [x,y,z] coordinate.
        v1: [x,y,z] coordinate of one of the triangle's vertices.
        v2: [x,y,z] coordinate of one of the triangle's vertices.
        v3: [x,y,z] coordinate of one of the triangle's vertices.

    Returns:
        True if p lies in the triangle, otherwise False.
    """

    n = np.cross(v2 - v1, v3 - v1)

    alpha = np.dot(n, np.cross(v3 - v2, p - v2)) / np.dot(n, n)
    beta = np.dot(n, np.cross(v1 - v3, p - v3)) / np.dot(n, n)
    gamma = np.dot(n, np.cross(v2 - v1, p - v1)) / np.dot(n, n)

    return (alpha >= 0.0) and (alpha <= 1.0) and (beta >= 0.0) and (beta <= 1.0) and (gamma >= 0.0) and (gamma <= 1.0)


def calc_unit_normal(v1: ndarray, v2: ndarray, v3: ndarray) -> ndarray:
    """
    Calculates the unit normal of the triangle v1-v2-v3.

    Args:
        v1: [x,y,z] coordinate of one of the triangle's vertices.
        v2: [x,y,z] coordinate of one of the triangle's vertices.
        v3: [x,y,z] coordinate of one of the triangle's vertices.

    Returns:
        Unit normal vector.
    """

    n = np.cross(v2 - v1, v3 - v1)
    n /= np.sqrt(np.dot(n, n))

    return n


def ray_trace_2d(x: float, y: float, polygon: List) -> bool:
    """
    Determine if the given 2D point is located inside the given polygon using the ray-tracing point-in-polygon
    technique.

    Extend a ray from the point along the positive x-axis to infinity and count the number of times the ray intersects
    the polygon. The point is located inside the polygon if there are an odd number of intersections.

    Args:
        x: x-coordinate of point.
        y: y-coordinate of point.
        polygon: List of coordinates of polygon vertices in counterclockwise order.

    Returns:
        Whether the point is inside the polygon.
    """

    n = len(polygon)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_new_bounds(bounds_lst: List, N: List[int], scale: List[float], offset: List[float]) \
        -> Tuple[List[int], List[float], List[float]]:
    """
    If a .stl file's boundaries exceed the bounds of the nonconformal mesh, find new bounds for generating the phase
    fields and masks.

    Args:
        bounds_lst: List containing lists of the min/max values in each direction (x,y or x,y,z).
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction.
        offset: Centers the meshed domain in each direction.

    Returns:
        Tuple[List[int], List[float], List[float]]:
            - tmp_N: Number of mesh elements to use when constructing the phase fields (preserves original dx).
            - tmp_scale: Scale to use when constructing the phase fields.
            - tmp_offset: Offset to use when constructing the phase fields.
    """

    for i in range(len(bounds_lst)):
        # Pad the bounds a bit so the boundary-mapping algorithm doesn't fail.
        bound_scale = bounds_lst[i][1] - bounds_lst[i][0]
        bounds_lst[i][0] -= 0.25 * bound_scale
        bounds_lst[i][1] += 0.25 * bound_scale

    # Get the new scale and offset.
    tmp_scale = [bound[1] - bound[0] for bound in bounds_lst]
    tmp_offset = [-bound[0] for bound in bounds_lst]

    # Get a new N that keeps the same mesh spacing.
    tmp_N = [int(tmp_scale[i] * N[i] / scale[i]) for i in range(len(N))]

    return tmp_N, tmp_scale, tmp_offset


def crop_to_mesh_bounds(arr: ndarray, N: List[int], scale: List[float], offset: List[float], tmp_N: List[int],
                        tmp_scale: List[float], tmp_offset: List[float]) -> ndarray:
    """
    Take an array that exceeds the nonconformal mesh's boundary and crop it so it fits within an array defined over the
    nonconformal mesh.

    Args:
        arr: The numpy array to crop.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction.
        offset: Centers the meshed domain in each direction.
        tmp_N: Number of mesh elements used when constructing the phase fields (preserves original dx).
        tmp_scale: Scale used when constructing the phase fields.
        tmp_offset: Offset used when constructing the phase fields.

    Returns:
        Numpy array covering only the bounds of the mesh and containing portions of arr.
    """

    shape = tuple([n+1 for n in N])
    fitted_arr = np.zeros(shape)

    # Determine the intersections of arr and fitted_arr along each direction.
    intersection_lst = [[max(-offset[i], -tmp_offset[i]), min(scale[i] - offset[i], tmp_scale[i] - tmp_offset[i])] for i in range(len(scale))]

    # Determine the arr indices corresponding to these intersections.
    arr_indices = []
    fitted_arr_indices = []
    for i in range(len(intersection_lst)):
        x = -offset[0] + scale[0] * i / N[0]
        min_fitted_arr_index = int((intersection_lst[i][0] + offset[i]) * N[i] / scale[i])
        max_fitted_arr_index = int((intersection_lst[i][1] + offset[i]) * N[i] / scale[i])

        min_arr_index = int((intersection_lst[i][0] + tmp_offset[i]) * tmp_N[i] / tmp_scale[i])
        max_arr_index = int((intersection_lst[i][1] + tmp_offset[i]) * tmp_N[i] / tmp_scale[i])

        # Confirm that the same number of elements are covered for both arrays.
        fitted_arr_delta = max_fitted_arr_index - min_fitted_arr_index
        arr_delta = max_arr_index - min_arr_index

        if fitted_arr_delta > arr_delta:
            max_fitted_arr_index = min_fitted_arr_index + arr_delta
        elif fitted_arr_delta < arr_delta:
            max_arr_index = min_arr_index + fitted_arr_delta

        arr_indices.append([min_arr_index, max_arr_index])
        fitted_arr_indices.append([min_fitted_arr_index, max_fitted_arr_index])

    # Replace sections of fitted_arr with sections of arr.
    if len(N) == 2:
        fitted_arr[fitted_arr_indices[0][0]:fitted_arr_indices[0][1]+1,
        fitted_arr_indices[1][0]:fitted_arr_indices[1][1]+1] \
            = arr[arr_indices[0][0]:arr_indices[0][1]+1, arr_indices[1][0]:arr_indices[1][1]+1]
    elif len(N) == 3:
        fitted_arr[fitted_arr_indices[0][0]:fitted_arr_indices[0][1]+1,
        fitted_arr_indices[1][0]:fitted_arr_indices[1][1]+1,
        fitted_arr_indices[2][0]:fitted_arr_indices[2][1]+1] \
            = arr[arr_indices[0][0]:arr_indices[0][1]+1, arr_indices[1][0]:arr_indices[1][1]+1,
              arr_indices[2][0]:arr_indices[2][1]+1]

    return fitted_arr


def get_stl_faces(filename: str) -> Tuple[ndarray, List]:
    """
    Compile the face vertices and outwards facing normals of a mesh defined in a .stl file into a list.

    The mesh will typically be a boundary mesh so face_lst can be used by ray_trace_3d.

    Args:
        filename: Path to the .stl file.

    Returns:
        Tuple[ndarray, List]:
            - face_lst: Numpy array listing out the vertices and outwards facing normals of the mesh's faces.
            - bounds_lst: List of lists of min and max values of the .stl file boundary vertices in each direction.
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError('The given .stl file does not exist.')

    if filename[-4:] != '.stl':
        raise TypeError('Expecting a .stl file.')

    with open(filename, 'r') as f:
        data = f.readlines()

    if 'solid' in data[0]:
        del data[0]

    if 'endsolid' in data[-1]:
        del data[-1]

    data_lst = []
    for line in data:
        if line.strip() and (('facet' in line) or ('vertex' in line) or ('loop' in line)):
            data_lst.append(line)

    face_lst = np.empty((len(data_lst) // 7, 12))
    x_bounds: List = []
    y_bounds: List = []
    z_bounds: List = []
    for i in range(0, len(data_lst), 7):
        # Assumes that the .stl file was generated from a mesh with all
        # outwards facing surface normals.
        n = np.array([float(num) for num in data_lst[i].split()[2:5]]) * (-1.0)
        v1 = np.array([float(num) for num in data_lst[i + 2].split()[1:4]])
        v2 = np.array([float(num) for num in data_lst[i + 3].split()[1:4]])
        v3 = np.array([float(num) for num in data_lst[i + 4].split()[1:4]])

        face_lst[i // 7, :3] = n
        face_lst[i // 7, 3:6] = v1
        face_lst[i // 7, 6:9] = v2
        face_lst[i // 7, 9:] = v3

        # Check for new min/max x-value.
        tmp_min_x = min(v1[0], min(v2[0], v3[0]))
        tmp_max_x = max(v1[0], max(v2[0], v3[0]))

        if not x_bounds:
            x_bounds = [tmp_min_x, tmp_max_x]
        else:
            if tmp_min_x < x_bounds[0]:
                x_bounds[0] = tmp_min_x
            elif tmp_max_x > x_bounds[1]:
                x_bounds[1] = tmp_max_x

        # Check for new min/max y-value.
        tmp_min_y = min(v1[1], min(v2[1], v3[1]))
        tmp_max_y = max(v1[1], max(v2[1], v3[1]))

        if not y_bounds:
            y_bounds = [tmp_min_y, tmp_max_y]
        else:
            if tmp_min_y < y_bounds[0]:
                y_bounds[0] = tmp_min_y
            elif tmp_max_y > y_bounds[1]:
                y_bounds[1] = tmp_max_y

        # Check for new min/max z-value.
        tmp_min_z = min(v1[2], min(v2[2], v3[2]))
        tmp_max_z = max(v1[2], max(v2[2], v3[2]))

        if not z_bounds:
            z_bounds = [tmp_min_z, tmp_max_z]
        else:
            if tmp_min_z < z_bounds[0]:
                z_bounds[0] = tmp_min_z
            elif tmp_max_z > z_bounds[1]:
                z_bounds[1] = tmp_max_z

    return face_lst, [x_bounds, y_bounds, z_bounds]


def get_Netgen_nonconformal(N: List[int], scale: List[float], offset: List[float], dim: int = 2, quad: bool = True) -> ngmsh.Mesh:
    """
    Generate a structured Netgen mesh over a prescribed square/cubic domain.

    Args:
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        dim: Dimension of the domain (must be 2 or 3).
        quad: If True, the mesh consists of quadrilateral/hexahedral elements. Otherwise the mesh 
            consists of triangular/tetrahedral elements.

    Returns:
        Structured Netgen mesh.
    """
    # Construct a Netgen mesh.
    ngmesh = ngmsh.Mesh()

    if dim == 2:
        ngmesh.dim = 2
        
        # N sets the number of nodes, so to get the expected number of mesh elements for triangular
        # meshes, N must be modified. Otherwise you would end up with 4*N^2 elements instead of N^2 
        # elements.
        if not quad:
            N = [round(n/2) for n in N]

        # Set evenly spaced mesh nodes.
        points = []
        coords = []
        for i in range(N[1] + 1):
            for j in range(N[0] + 1):
                x = -offset[0] + scale[0] * j / N[0]
                y = -offset[1] + scale[1] * i / N[1]
                z = 0.0
                points.append(ngmesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(x, y, z))))
                coords.append((x, y, z))

        # TODO: Should the user be able to set their own BC names?
        idx_dom = ngmesh.AddRegion('dom', dim=2)
        idx_bottom = ngmesh.AddRegion('bottom', dim=1)
        idx_right = ngmesh.AddRegion('right', dim=1)
        idx_top = ngmesh.AddRegion('top', dim=1)
        idx_left = ngmesh.AddRegion('left', dim=1)

        # Generate mesh faces.
        for i in range(N[1]):
            for j in range(N[0]):
                p1 = i * (N[0] + 1) + j
                p2 = i * (N[0] + 1) + j + 1
                p3 = i * (N[0] + 1) + j + 2 + N[0]
                p4 = i * (N[0] + 1) + j + 1 + N[0]
                
                if quad:
                    # Add one quadrilateral element.
                    ngmesh.Add(ngmsh.Element2D(idx_dom, [points[p1], points[p2], points[p3], points[p4]]))
                else:
                    # Need one additional point in the center of the square.
                    x = 0.5 * (coords[p1][0] + coords[p2][0])
                    y = 0.5 * (coords[p1][1] + coords[p3][1])
                    z = 0
                    center_point = ngmesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(x, y, z)))
                        
                    # Add four triangular elements by quartering the quadrilateral element.
                    ngmesh.Add(ngmsh.Element2D(idx_dom, [points[p1], points[p2], center_point]))
                    ngmesh.Add(ngmsh.Element2D(idx_dom, [points[p2], points[p3], center_point]))
                    ngmesh.Add(ngmsh.Element2D(idx_dom, [points[p3], points[p4], center_point]))
                    ngmesh.Add(ngmsh.Element2D(idx_dom, [points[p4], points[p1], center_point]))

        # Assign each edge of the domain to the same boundary.
        for i in range(N[1]):
            ngmesh.Add(ngmsh.Element1D([points[N[0] + i * (N[0] + 1)], points[N[0] + (i + 1) * (N[0] + 1)]], index=idx_right))
            ngmesh.Add(ngmsh.Element1D([points[(i + 1) * (N[0] + 1)], points[i * (N[0] + 1)]], index=idx_left))
            
        for i in range(N[0]):
            ngmesh.Add(ngmsh.Element1D([points[i], points[i + 1]], index=idx_bottom))
            ngmesh.Add(ngmsh.Element1D([points[1 + i + N[1] * (N[0] + 1)], points[i + N[1] * (N[0] + 1)]], index=idx_top))
            
    elif dim == 3:
        ngmesh.dim = 3

        # N sets the number of nodes, so to get the expected number of mesh elements for tetrahedral
        # meshes, N must be modified. Otherwise you would end up with 6*N^3 elements instead of N^3 
        # elements.
        if not quad:
            N = [round(n/6**(1/3)) for n in N]

        # Set evenly spaced mesh nodes.
        points = []
        coords = []
        for i in range(N[0] + 1):
            for j in range(N[1] + 1):
                for k in range(N[2] + 1):
                    x = -offset[0] + scale[0] * i / N[0]
                    y = -offset[1] + scale[1] * j / N[1]
                    z = -offset[2] + scale[2] * k / N[2]
                    points.append(ngmesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(x, y, z))))
                    coords.append((x, y, z))

        # Generate mesh cells.
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    base = i * (N[1] + 1) * (N[2] + 1) + j * (N[2] + 1) + k
                    baseup = base + (N[1] + 1) * (N[2] + 1)
                    p1 = base
                    p2 = base + 1
                    p3 = base + (N[2] + 1) + 1
                    p4 = base + (N[2] + 1)
                    p5 = baseup
                    p6 = baseup + 1
                    p7 = baseup + (N[2] + 1) + 1
                    p8 = baseup + (N[2] + 1)
                    idx = 1
                    
                    if quad:
                        # Add one hexahedral element.
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p1], points[p2], points[p3], points[p4], points[p5], points[p6],
                                              points[p7], points[p8]]))
                    else:
                        # Need one additional point in the center of the cube.
                        x = 0.5 * (coords[p1][0] + coords[p5][0])
                        y = 0.5 * (coords[p1][1] + coords[p4][1])
                        z = 0.5 * (coords[p1][2] + coords[p2][2])
                        center_point = ngmesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(x, y, z)))
                        
                        # Add six tetrahedral elements.
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p1], points[p5], center_point, points[p8], points[p4]]))
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p5], points[p6], center_point, points[p7], points[p8]]))
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p6], points[p2], center_point, points[p3], points[p7]]))
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p2], points[p1], center_point, points[p4], points[p3]]))
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p2], points[p6], center_point, points[p5], points[p1]]))
                        ngmesh.Add(ngmsh.Element3D(idx,
                                             [points[p4], points[p8], center_point, points[p7], points[p3]]))

        def add_bc(p, d, N, deta, neta, facenr):

            def add_seg(i, j, os):
                base = p + i * d + j * deta
                p1 = base
                p2 = base + os
                ngmesh.Add(ngmsh.Element1D([points[p1], points[p2]], index=facenr))

                return

            for i in range(N):
                for j in [0, neta]:
                    add_seg(i, j, d)

            for i in [0, N]:
                for j in range(neta):
                    add_seg(i, j, deta)

            for i in range(N):
                for j in range(neta):
                    base = p + i * d + j * deta
                    p1 = base
                    p2 = base + d
                    p3 = base + d + deta
                    p4 = base + deta
                    ngmesh.Add(ngmsh.Element2D(facenr, [points[p1], points[p2], points[p3], points[p4]]))

            return

        # Order is important!
        ngmesh.Add(ngmsh.FaceDescriptor(surfnr=4, domin=1, bc=1))
        ngmesh.Add(ngmsh.FaceDescriptor(surfnr=2, domin=1, bc=2))
        ngmesh.Add(ngmsh.FaceDescriptor(surfnr=5, domin=1, bc=3))
        ngmesh.Add(ngmsh.FaceDescriptor(surfnr=3, domin=1, bc=4))
        ngmesh.Add(ngmsh.FaceDescriptor(surfnr=0, domin=1, bc=5))
        ngmesh.Add(ngmsh.FaceDescriptor(surfnr=1, domin=1, bc=6))

        # Assign each exterior face of the domain to its respective boundary.
        add_bc(0, 1, N[2], N[2] + 1, N[1], 1)
        add_bc(0, (N[1] + 1) * (N[2] + 1), N[0], 1, N[2], 2)
        add_bc((N[0] + 1) * (N[1] + 1) * (N[2] + 1) - 1, -(N[2] + 1), N[1], -1, N[2], 3)
        add_bc((N[0] + 1) * (N[1] + 1) * (N[2] + 1) - 1, -1, N[2], -(N[1] + 1) * (N[2] + 1), N[0], 4)
        add_bc(0, N[2] + 1, N[1], (N[1] + 1) * (N[2] + 1), N[0], 5)
        add_bc((N[0] + 1) * (N[1] + 1) * (N[2] + 1) - 1, -(N[1] + 1) * (N[2] + 1), N[0], -(N[2] + 1), N[1], 6)

        # TODO: Should the user be able to specify their own bc names?
        bc_names = {0: 'back', 1: 'left', 2: 'front', 3: 'right', 4: 'bottom', 5: 'top'}
        for key, val in bc_names.items():
            ngmesh.SetBCName(key, val)

    else:
        raise ValueError('Only works with 2D or 3D meshes.')

    ngmesh.Compress()
    
    return ngmesh

def get_mesh_boundary_from_conformal_2d(mesh: ngmsh.Mesh) -> List:
    """
    Get the ordered boundary vertices of a 2D Netgen mesh.

    Args:
        mesh: 2D Netgen mesh to find the boundary vertices of.

    Returns:
        List of coordinates of Netgen mesh boundary vertices in counterclockwise order.
    """

    edge_lst = []
    vertex_lst = []
    vertex_coords = []
    for edge in mesh.Elements1D():
        edge_lst.append(edge.index)
        tmp_lst = []
        tmp_coords = []
        for vertex in edge.vertices:
            tmp_lst.append(vertex)
            tmp_coords.append((mesh[vertex][0], mesh[vertex][1]))
        vertex_lst.append(tmp_lst)
        vertex_coords.append(tmp_coords)

    boundary_lst = reorder_vertices_2d(edge_lst, vertex_lst, vertex_coords)
    boundary_lst = order_ccw(boundary_lst)

    return boundary_lst


def get_mesh_boundary_2d(filename: str) -> Tuple[List, List]:
    """
    Get the ordered boundary vertices of a 2D .stl file.

    Args:
        filename: Path to .stl file.

    Returns:
        Tuple[List, List]:
            - boundary_lst: List of coordinates of .stl file boundary vertices in counterclockwise order.
            - bounds_lst: List of lists of min and max values of the .stl file boundary vertices in each direction.
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError('The given .stl file does not exist.')

    if filename[-4:] != '.stl':
        raise TypeError('Expecting a .stl file.')

    with open(filename, 'r') as f:
        data = f.readlines()

    data_lst = []
    x_bounds: List = []
    y_bounds: List = []
    for line in data:
        if line.strip() and ('vertex' in line):
            v = tuple([float(num) for num in line.split()[1:3]])
            data_lst.append(v)

            # Check for new min/max x-value.
            if not x_bounds:
                x_bounds = [v[0], v[0]]
            else:
                if v[0] < x_bounds[0]:
                    x_bounds[0] = v[0]
                elif v[0] > x_bounds[1]:
                    x_bounds[1] = v[0]

            # Check for new min/max y-value.
            if not y_bounds:
                y_bounds = [v[1], v[1]]
            else:
                if v[1] < y_bounds[0]:
                    y_bounds[0] = v[1]
                elif v[1] > y_bounds[1]:
                    y_bounds[1] = v[1]

    all_v_pairs = []
    for i in range(0, len(data_lst), 3):
        v1 = tuple(data_lst[i])
        v2 = tuple(data_lst[i + 1])
        v3 = tuple(data_lst[i + 2])

        # Need to use sets so that [A,B] and [B,A] are treated as the same
        # vertex pair.
        all_v_pairs.append(set([v1, v2]))
        all_v_pairs.append(set([v2, v3]))
        all_v_pairs.append(set([v1, v3]))

    vertex_coords = []
    vertex_lst = []
    edge_lst = []
    idx = 0
    for item in all_v_pairs:
        if all_v_pairs.count(item) == 1:
            v1, v2 = item
            v1_num = data_lst.index(v1)
            v2_num = data_lst.index(v2)

            vertex_coords.append([v1, v2])
            vertex_lst.append([v1_num, v2_num])
            edge_lst.append(idx)

            idx += 1

    boundary_lst = reorder_vertices_2d(edge_lst, vertex_lst, vertex_coords)
    boundary_lst = order_ccw(boundary_lst)

    return boundary_lst, [x_bounds, y_bounds]


def get_mesh_boundary_3d(filename: str) -> Tuple[List, List]:
    """
    Get the edges and boundary edges of a 3D .msh or .stl file.

    This function should only be used for surface meshes since the boundary edges are considered to be edges with only
    one face. Volume meshes do not have such boundary edges. NOTE: Using a .msh file does not always give a
    comprehensive edge_lst since for some unknown reason ngmesh.Elements1D() does not always give all of the edges in
    the mesh. This is not necessarily a problem if edge_lst is only being used to produce BC masks (but always check!).
    However, using a .stl file should always give the full boundary_lst and edge_lst.

    Args:
        filename: Path to .msh or .stl file.

    Returns:
        Tuple[List, List]:
            - edge_lst: List of coordinates of edge vertices.
            - boundary_lst: List of coordinates of boundary edge vertices.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError('The given mesh file does not exist.')

    if filename[-4:] == '.msh':
        ngmesh = ReadGmsh(filename)

        data_lst = []
        for face in ngmesh.Elements2D():
            data_lst.append(face.vertices)

        all_v_pairs = []
        for item in data_lst:
            v1_num, v2_num, v3_num = item
            all_v_pairs.append(set([v1_num, v2_num]))
            all_v_pairs.append(set([v2_num, v3_num]))
            all_v_pairs.append(set([v1_num, v3_num]))

        boundary_lst = []
        for item in all_v_pairs:
            v1_num, v2_num = item
            v1 = list(ngmesh[v1_num])
            v2 = list(ngmesh[v2_num])

            if all_v_pairs.count(item) == 1:
                boundary_lst.append(v1 + v2)

        edge_lst = []
        for edge in ngmesh.Elements1D():
            v1_num, v2_num = edge.vertices
            v1 = list(ngmesh[v1_num])
            v2 = list(ngmesh[v2_num])
            edge_lst.append(v1 + v2)

    elif filename[-4:] == '.stl':
        face_lst = get_stl_faces(filename)

        all_v_pairs = []
        for item in face_lst:
            v1_tup = (item[3], item[4], item[5])
            v2_tup = (item[6], item[7], item[8])
            v3_tup = (item[9], item[10], item[11])

            all_v_pairs.append(set([v1_tup, v2_tup]))
            all_v_pairs.append(set([v2_tup, v3_tup]))
            all_v_pairs.append(set([v1_tup, v3_tup]))

        # all_v_pairs contains a separate instance of each edge for each face 
        # that contains it. edge_lst should only contain one instance of each 
        # edge and boundary_lst should only contain one instance of each 
        # boundary edge.
        boundary_lst = []
        edge_lst = []
        for item in all_v_pairs:
            v1, v2 = item

        new_item = list(v1) + list(v2)

        if all_v_pairs.count(item) == 1:
            boundary_lst.append(new_item)
            edge_lst.append(new_item)
        elif new_item in edge_lst:
            pass
        else:
            edge_lst.append(new_item)

    else:
        raise TypeError('Expecting a .msh file or a .stl file.')

    return edge_lst, boundary_lst


def get_mesh_edges_vertices_3d(filename: str) -> Tuple[ndarray, List, List, List]:
    """
    Get the edges and vertices of a 3D .stl file.

    This is a helper function for calculating mesh quality metrics.

    Args:
        filename: Path to the .stl file.

    Returns:
        Tuple[List, List, List, List]:
            - face_lst: List of the vertices and outwards facing normals of the mesh's faces.
            - edge_lst: List of coordinates of edge vertices.
            - v_lst: List of coordinates of vertices.
            - v_con_lst: List containing each vertex and an ordered list of its neighbouring vertices (coordinate form).
    """

    face_lst = get_stl_faces(filename)

    all_v_pairs = []
    v_set = set()
    for item in face_lst:
        v1 = (item[3], item[4], item[5])
        v2 = (item[6], item[7], item[8])
        v3 = (item[9], item[10], item[11])

        v_set.add(v1)
        v_set.add(v2)
        v_set.add(v3)

        all_v_pairs.append(set([v1, v2]))
        all_v_pairs.append(set([v2, v3]))
        all_v_pairs.append(set([v1, v3]))

    v_lst = list(v_set)

    edge_lst = []
    for item in all_v_pairs:
        if item in edge_lst:
            pass
        else:
            edge_lst.append(item)

    edge_lst = [list(item) for item in edge_lst]

    v_con_lst = []
    for v in v_lst:

        con_lst = []
        for item in edge_lst:
            v1, v2 = item
            if v1 == v:
                con_lst.append(v2)
            elif v2 == v:
                con_lst.append(v1)
            else:
                pass

        con_lst_prime = [con_lst.pop(0)]
        for i in range(len(con_lst)):
            for con in con_lst:
                if set([con_lst_prime[-1], con]) in edge_lst:
                    con_lst_prime.append(con)
                    con_lst.remove(con)
                else:
                    pass

        v_con_lst.append([v, con_lst_prime])

    return face_lst, edge_lst, v_lst, v_con_lst
