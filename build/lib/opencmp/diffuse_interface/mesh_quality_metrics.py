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
import scipy.optimize as spopt
from . import mesh_helpers
from typing import List, Tuple


def line_segments_intersect_2d(p1: List[float], p2: List[float], p3: List[float], p4: List[float]) -> bool:
    """
    Check if two 2D line segments intersect.

    Args:
        p1: [x,y] coordinate of a boundary point of the first line segment.
        p2: [x,y] coordinate of a boundary point of the first line segment.
        p3: [x,y] coordinate of a boundary point of the second line segment.
        p4: [x,y] coordinate of a boundary point of the second line segment.

    Returns:
        True if the line segments intersect, otherwise False.
    """

    order_123 = mesh_helpers.orient_2d(p1, p2, p3)
    order_124 = mesh_helpers.orient_2d(p1, p2, p4)
    order_341 = mesh_helpers.orient_2d(p3, p4, p1)
    order_342 = mesh_helpers.orient_2d(p3, p4, p2)

    if (order_123 != order_124) and (order_341 != order_342):
        # The line segments intersect.
        return True
    elif (order_123 == 'collinear'):
        # p1, p2, and p3 are collinear. Check if p3 lies on p1-p2.
        if (p3[0] <= max(p1[0], p2[0])) and (p3[0] >= min(p1[0], p2[0])) and (p3[1] <= max(p1[1], p2[1])) and (
                p3[1] >= max(p1[1], p2[1])):
            return True
    elif (order_124 == 'collinear'):
        # p1, p2, and p4 are collinear. Check if p4 lies on p1-p2.
        if (p4[0] <= max(p1[0], p2[0])) and (p4[0] >= min(p1[0], p2[0])) and (p4[1] <= max(p1[1], p2[1])) and (
                p4[1] >= max(p1[1], p2[1])):
            return True
    elif (order_341 == 'collinear'):
        # p3, p4, and p1 are collinear. Check if p1 lies on p3-p4.
        if (p1[0] <= max(p3[0], p4[0])) and (p1[0] >= min(p3[0], p4[0])) and (p1[1] <= max(p3[1], p4[1])) and (
                p1[1] >= max(p3[1], p4[1])):
            return True
    elif (order_342 == 'collinear'):
        # p3, p4, and p2 are collinear. Check if p2 lies on p3-p4.
        if (p2[0] <= max(p3[0], p4[0])) and (p2[0] >= min(p3[0], p4[0])) and (p2[1] <= max(p3[1], p4[1])) and (
                p2[1] >= max(p3[1], p4[1])):
            return True

    # The line segments do not intersect.
    return False


def line_segment_face_intersect_3d(p1_tmp: List[float], p2_tmp: List[float], v1_tmp: List[float], v2_tmp: List[float],
                                   v3_tmp: List[float], n_tmp: List[float]) -> bool:
    """
    Check if line segment p1-p2 intersects face v1-v2-v3.

    Args:
        p1_tmp: [x,y,z] coordinate of a boundary point of the line segment.
        p2_tmp: [x,y,z] coordinate of a boundary point of the line segment.
        v1_tmp: [x,y,z] coordinate of a vertex of the face.
        v2_tmp: [x,y,z] coordinate of a vertex of the face.
        v3_tmp: [x,y,z] coordinate of a vertex of the face.

    Returns:
        True if the line segment intersects the face, otherwise False.
    """

    p1 = np.array(p1_tmp)
    p2 = np.array(p2_tmp)
    v1 = np.array(v1_tmp)
    v2 = np.array(v2_tmp)
    v3 = np.array(v3_tmp)
    n = np.array(n_tmp)

    # Use signed volumes to check if the line segment intersects the plane of 
    # the face or not. If either endpoint lies on the plane of the face, use 
    # barycentric coordinates to determine if that endpoint lies within the 
    # face.
    s1 = mesh_helpers.orient_3d(p1, v1, v2, v3)
    s2 = mesh_helpers.orient_3d(p2, v1, v2, v3)

    plane1 = (s1 == 0.0)
    face1 = mesh_helpers.calc_barycentric(p1, v1, v2, v3)

    plane2 = (s2 == 0.0)
    face2 = mesh_helpers.calc_barycentric(p2, v1, v2, v3)

    if plane1 or plane2:
        # At least one endpoint lies on the plane of the face.
        if plane1 * face1 or plane2 * face2:
            # At least one endpoint lies on the plane of the face and within 
            # the face.
            return True
        else:
            # At least one endpoint lies on the plane of the face but not 
            # within the face.
            return False
    elif (s1 == s2):
        # Both endpoints lie on the same side as the plane. There are no 
        # intersections.
        return False
    else:
        # The endpoints lie on opposite sides of the plane of the face. Check 
        # if the intersection point lies within the face.
        t = np.dot(n, v1 - p1) / np.dot(n, p2 - p1)
        v4 = p1 + (p2 - p1) * t

        # If the intersection point lies within the face there is an 
        # intersection, otherwise there is no intersection.
        return mesh_helpers.calc_barycentric(v4, v1, v2, v3)


def calc_curvature_3d(v: ndarray, con_lst: List) -> float:
    """
    Estimates the mean curvature at a point by fitting an osculating parabola to the 3D surface surrounding the point.

    !!! NOT WORKING PROPERLY !!!

    Args:
        v: [x, y, z] coordinates of the vertex of interest.
        con_lst: A list of the coordinates of all the vertices that share an edge with v. The vertices are ordered by
            their shared edges.

    Returns:
        Estimate of the mean curvature at v.
    """

    # Approximate the surface unit normal as the average of the unit normals of the faces connected to v.
    n = mesh_helpers.calc_unit_normal(v, con_lst[-1], con_lst[0])
    for i in range(len(con_lst) - 1):
        n += mesh_helpers.calc_unit_normal(v, con_lst[i], con_lst[i + 1])
    n /= len(con_lst)

    # v_arr will hold v and all of the connecting vertices rotated such that n is parallel to the z-axis and translated
    # such that v lies on z=0.
    v_arr = np.empty((len(con_lst) + 1, 3))

    if np.allclose(n, np.array([0.0, 0.0, 1.0])):
        # n is parallel to the z-axis. Only need to translate the vertices.
        T = np.array([0.0, 0.0, -v[2]])

        v_arr[0, :] = v + T
        v_arr[1:, :] = con_lst + T

    elif np.allclose(n, np.array([0.0, 0.0, -1.0])):
        # n is parallel to the negative z-axis. Reflect the vertices and then translate them.
        v_arr[0, :] = v * np.array([1.0, 1.0, -1.0])
        v_arr[1:, :] = con_lst * np.array([1.0, 1.0, -1.0])

        T = np.array([0.0, 0.0, -v_arr[0, 2]])
        v_arr += T

    else:
        # n is not collinear with the z-axis. Rotate and translate the vertices.
        r = np.cross(n, np.array([0.0, 0.0, 1.0]))
        c = np.dot(n, np.array([0.0, 0.0, 1.0]))
        rx = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        R = np.identity(3) + rx + np.matmul(rx, rx) * (1.0 / (1.0 + c))

        v_arr[0, :] = np.matmul(v, R)
        v_arr[1:, :] = np.array([np.matmul(con, R) for con in con_lst])

        T = np.array([0.0, 0.0, -v_arr[0, 2]])
        v_arr += T

    # Fit the vertices to an osculating parabola to find the mean curvature.
    quad_func = lambda p, a, b, c: a * p[:, 0] ** 2 + b * p[:, 0] * p[:, 1] + c * p[:, 1] ** 2
    a, b, c = spopt.curve_fit(quad_func, v_arr[:, :2], v_arr[:, 2])[0]

    return abs(a + c)


def get_chords_2d(boundary_lst: List, crossing: bool = True, separation: int = 0) -> List:
    """
    Get every chord length in a 2D polygon.

    A chord length is defined as the Euclidean distance between two different polygon vertices. Chords can be excluded
    if they cross the polygon's boundary or if their bounding vertices are too close to each other.

    Args:
        boundary_lst: List of coordinates of the polygon boundary vertices in counterclockwise order.
        crossing: If False, only chords completely contained within the polygon's boundary are included.
        separation: The minimum number of vertices that must separate a pair of boundary vertices in order to have a
            chord between those vertices. This can be used to find necks.

    Returns:
        List of all of the polygon's chord lengths.
    """

    if (boundary_lst[0] == boundary_lst[-1]):
        boundary_lst.pop(-1)

    chords = []

    for i in range(len(boundary_lst) - separation):
        for j in range(i - separation):
            v1 = boundary_lst[i]
            v2 = boundary_lst[j]

            if not crossing:
                # Only allow chords that are fully contained within the 
                # polygon's boundary.
                for k in range(len(boundary_lst) - 1):
                    v3 = boundary_lst[k]
                    v4 = boundary_lst[k + 1]

                    if (k == i) or (k + 1 == i) or (k == j) or (k + 1 == j):
                        # Ignore any boundary edges connected to v1 or v2. They 
                        # don't count as intersections.
                        pass
                    elif line_segments_intersect_2d(v1, v2, v3, v4):
                        # Don't include the chord if it intersects any of the 
                        # other boundary edges.
                        break
                else:
                    chord_len = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
                    chords.append(chord_len)

            else:
                # Allow chords that intersect the polygon's boundary.
                chord_len = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
                chords.append(chord_len)

    # Split up the range so separation wraps properly.         
    for i in range(len(boundary_lst) - separation, len(boundary_lst)):
        for j in range(separation - (len(boundary_lst) - i - 1), i - separation):
            v1 = boundary_lst[i]
            v2 = boundary_lst[j]

            if not crossing:
                # Only allow chords that are fully contained within the 
                # polygon's boundary.
                for k in range(len(boundary_lst) - 1):
                    v3 = boundary_lst[k]
                    v4 = boundary_lst[k + 1]

                    if (k == i) or (k + 1 == i) or (k == j) or (k + 1 == j):
                        # Ignore any boundary edges connected to v1 or v2. They 
                        # don't count as intersections.
                        pass
                    elif line_segments_intersect_2d(v1, v2, v3, v4):
                        # Don't include the chord if it intersects any of the 
                        # other boundary edges.
                        break
                else:
                    chord_len = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
                    chords.append(chord_len)

            else:
                # Allow chords that intersect the polygon's boundary.
                chord_len = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
                chords.append(chord_len)

    return chords


def get_radius_curvature_2d(boundary_lst: List) -> Tuple[List[float], List[float]]:
    """
    Get the radius of curvature at every boundary vertex in a 2D polygon.

    Exclude vertices whose connected edges make a 90 degree or acute angle. List these instead as discontinuities with
    magnitudes equal to the inverse of their angle.

    Args:
        boundary_lst: List of coordinates of the polygon boundary vertices in counterclockwise order.

    Returns:
        Tuple[List[float], List[float]]:
            - rc: List of all of the polygon's radii of curvature.
            - discont: List of the magnitudes of all of the polygon's discontinuities.
    """

    if boundary_lst[0] == boundary_lst[-1]:
        boundary_lst.pop(-1)

    rc = []
    discont = []

    for i in range(len(boundary_lst)):
        # Wrap correctly at the edfes of boundary_lst.
        if i == 0:
            v1 = boundary_lst[-1]
            v2 = boundary_lst[0]
            v3 = boundary_lst[1]
        elif i == len(boundary_lst) - 1:
            v1 = boundary_lst[-2]
            v2 = boundary_lst[-1]
            v3 = boundary_lst[0]
        else:
            v1 = boundary_lst[i - 1]
            v2 = boundary_lst[i]
            v3 = boundary_lst[i + 1]

        if mesh_helpers.orient_2d(v1, v2, v3) == 'collinear':
            # Straight lines have rc = inf.
            rc.append(np.inf)

        else:
            # The angle used to calculate the radius of curvature is the 
            # external angle.
            angle = mesh_helpers.angle_between(v1, v2, v3)
            ext_angle = abs(angle - np.pi)

            if ext_angle < np.pi / 2.0:
                l1 = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
                l2 = np.sqrt((v2[0] - v3[0]) ** 2 + (v2[1] - v3[1]) ** 2)

                rc_i = (l1 + l2) / (2.0 * ext_angle)
                rc.append(rc_i)

            else:
                # The point contains a discontinuity if the internal angle is 
                # an acute angle. The magnitude of the discontinuity is the
                # inverse of the angle.
                int_angle = np.pi - ext_angle

                if int_angle < 1.0e-10:
                    discont.append(np.inf)
                else:
                    l_discont = 1.0 / int_angle
                    discont.append(l_discont)

    return rc, discont


def get_chords_3d(boundary_lst: List, face_lst: List, crossing: bool = True) -> List:
    """
    Get every chord length in a 3D polygon.

    A chord length is defined as the Euclidean distance between two different polygon vertices. Chords can be excluded
    if they cross the polygon's boundary.

    Args:
        boundary_lst: List of coordinates of the polygon boundary vertices.
        face_lst: List of the vertices and outwards facing normals of the polygon's faces.
        crossing: If False, only chords completely contained within the polygon's boundary are included.

    Returns:
        List of all of the polygon's chord lengths.
    """

    chords = []

    for i in range(len(boundary_lst)):
        for j in range(i):
            v1 = boundary_lst[i]
            v2 = boundary_lst[j]

            if not crossing:
                # Only allow chords that are fully contained within the polygon's boundary.
                for face in face_lst:
                    n = face[:3]
                    v3 = face[3:6]
                    v4 = face[6:9]
                    v5 = face[9:12]

                    if (v3 == v1) or (v3 == v2) or (v4 == v1) or (v4 == v2) or (v5 == v1) or (v5 == v2):
                        # Ignore any boundary faces connected to v1 or v2. They don't count as intersections.
                        pass
                    elif line_segment_face_intersect_3d(v1, v2, v3, v4, v5, n):
                        # Don't include the chord if it intersects any of the other boundary faces.
                        break

                else:
                    chord_len = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)
                    chords.append(chord_len)

            else:
                # Allow chords that intersect the polygon's boundary.
                chord_len = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)
                chords.append(chord_len)

    return chords


def get_radius_curvature_3d(v_con_lst: List) -> Tuple[List[float], List[float]]:
    """
    Get the radius of curvature at every boundary vertex in a 3D polygon.

    | Exclude vertices where the boundary surface has a mean curvature of 90 degrees or less. List these instead as
    | discontinuities with magnitudes equal to the inverse of their mean curvature.
    |
    | !!! NOT WORKING PROPERLY !!!

    Args:
        v_con_lst: List containing each vertex and an ordered list of its neighbouring vertices (coordinate form).

    Returns:
        Tuple[List[float], List[float]]:
            - rc: List of all of the polygon's radii of curvature.
            - discont: List of the magnitudes of all of the polygon's discontinuities.
    """

    rc = []
    discont = []

    for item in v_con_lst:
        v, con_lst = item
        v = np.array(v)
        con_lst = np.array(con_lst)

        k_i = calc_curvature_3d(v, con_lst)

        if (k_i == 0.0):
            # Plane surfaces have rc = inf.
            rc.append(np.inf)
        else:
            rc_i = 1.0 / k_i
            dist = np.sqrt(2) * np.mean(
                np.sqrt((v[0] - con_lst[:, 0]) ** 2 + (v[1] - con_lst[:, 1]) ** 2 + (v[2] - con_lst[:, 2]) ** 2))

            if (rc_i <= dist):
                # The radius of curvature is smaller than the diagonal length of a corner made by the points.
                # This is considered a discontinuity whose magnitude is the factor by which the diagonal length exceeds
                # the radius of curvature.
                l_discont = dist / rc_i
                discont.append(l_discont)

            else:
                # There is no discontinuity.
                rc.append(rc_i)

    return rc, discont
