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

import ngsolve as ngs
from typing import List, Optional, Tuple, Callable
from ngsolve import CoefficientFunction, Mesh, GridFunction, Parameter
import numpy as np
from numpy import ndarray


def construct_identity_mat(dim: int) -> CoefficientFunction:
    """
    Constructs an identity matrix of the given dimension.

    This expects

    Args:
        dim: The dimension of the matrix.

    Returns:
        An identity matrix of the desired dimension.
    """
    if dim < 1:
        raise ValueError('Can\'t construct a {}D identity matrix.'.format(dim))

    lst = []
    for i in range(dim):
        new_lst = [0.0] * i + [1.0] + [0.0] * (dim - i - 1)
        lst += new_lst

    return ngs.CoefficientFunction(tuple(lst), dims=(dim, dim))


def get_special_functions(mesh: Mesh, nu: float) \
        -> Tuple[CoefficientFunction, CoefficientFunction, CoefficientFunction, CoefficientFunction]:
    """
    Generates a set of special functions needed for DG so they don't need to be rewritten multiple times.
    
    Args:
        mesh: The mesh used for the simulation.
        nu: The penalty parameter for interior penalty method DG.
        
    Returns:
        Tuple[CoefficientFunction, CoefficientFunction, CoefficientFunction, CoefficientFunction]:
            - n: The unit normal for every facet of the mesh.
            - h: The "size" of every mesh element.
            - alpha: The penalty coefficient.
            - I_mat: An identity matrix that matches the mesh dimension.
    """

    n = ngs.specialcf.normal(mesh.dim)
    h = ngs.specialcf.mesh_size
    alpha = nu / h
    I_mat = construct_identity_mat(mesh.dim)

    return n, h, alpha, I_mat


def ngsolve_to_numpy(mesh: Mesh, gfu: GridFunction, N: List[int], scale: List[float], offset: List[float], dim: int = 2,
                     binary: Optional[ndarray] = None) -> List[ndarray]:
    """
    Assign the values in gfu to a numpy array whose elements correspond to the nodes of mesh while preserving spatial
    ordering.

    The output array only contains node values, never the values of higher order dofs.

    Args:
        mesh: Structured quadrilateral/hexahedral NGSolve mesh.
        gfu: NGSolve GridFunction containing the values to assign.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        dim: Dimension of the domain (must be 2 or 3).
        binary: If exists, arr is multiplied by binary to zero out any array elements outside of
            the approximated complex geometry.

    Returns:
        List of arrays containing the values in gfu. Since gfu may be vector-valued, the arrays in arr_vec correspond to
        the various components of gfu.
    """

    N_arr = np.array(N) + 1
    arr = np.zeros(N_arr)

    # The GridFunction may be vector-valued.
    # Return a list containing the spatial field for each
    # vector component.
    arr_vec = [arr] * gfu.dim

    # TODO: combine these two branches of the if statement together (it's the same code).
    if dim == 2:
        for i in range(N[0] + 1):
            for j in range(N[1] + 1):
                x = -offset[0] + scale[0] * i / N[0]
                y = -offset[1] + scale[1] * j / N[1]

                mip = mesh(x, y)
                val = gfu(mip)

                if isinstance(val, tuple):
                    for k in range(len(val)):
                        arr_vec[k][i, j] = val[k]
                else:
                    arr_vec[0][i, j] = val

    elif dim == 3:
        for i in range(N[0] + 1):
            for j in range(N[1] + 1):
                for k in range(N[2] + 1):
                    x = -offset[0] + scale[0] * i / N[0]
                    y = -offset[1] + scale[1] * j / N[1]
                    z = -offset[2] + scale[2] * k / N[2]

                    mip = mesh(x, y, z)
                    val = gfu(mip)

                    if isinstance(val, tuple):
                        for m in range(len(val)):
                            arr_vec[m][i, j, k] = val[m]
                    else:
                        arr_vec[0][i, j, k] = val

    else:
        raise ValueError('`NGSolve_to_numpy` called with dimension of {}. '
                         'It only works with 2D or 3D meshes.'.format(dim))

    if binary is not None:
        for arr in arr_vec:
            arr *= binary

    return arr_vec


def numpy_to_ngsolve(mesh: Mesh, interp_ord: int, arr: ndarray, scale: List[float], offset: List[float], dim: int = 2) \
        -> GridFunction:
    """
    Assign the values in arr to a NGSolve GridFunction defined over the given NGSolve mesh while preserving spatial
    ordering.

    Args:
        mesh: Mesh used by the problem the GridFunction will be needed to solve.
        interp_ord: Interpolant order of the finite element space for the problem.
        arr: Array containing the values to assign.
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        dim: Dimension of the domain (must be 2 or 3).

    Returns:
        Gridfunction containing the values in arr.
    """

    # TODO: combine these two branches of the if statement together (it's the same code).
    if dim == 2:
        # Corrects for x,y mismatch.
        arr = np.transpose(arr)
        func = ngs.VoxelCoefficient((-offset[0], -offset[1]),
                                    (scale[0] - offset[0], scale[1] - offset[1]),
                                    arr,
                                    linear=True)
    elif dim == 3:
        # Corrects for x,y,z mismatch.
        arr = np.transpose(arr, (2, 1, 0))
        func = ngs.VoxelCoefficient((-offset[0], -offset[1], -offset[2]),
                                    (scale[0] - offset[0], scale[1] - offset[1], scale[2] - offset[2]),
                                    arr,
                                    linear=True)
    else:
        raise ValueError('`numpy_to_NGSolve` called with dimension of {}.'
                         'It only works with 2D or 3D meshes.'.format(dim))

    fes = ngs.H1(mesh, order=interp_ord)
    grid_func_tmp = ngs.GridFunction(fes)
    grid_func_tmp.Set(func)

    # Smooths out irregularities caused by the interpolation of the VoxelCoefficient onto the finite element space.
    grid_func = ngs.GridFunction(fes)
    grid_func.Set(ngs.Norm(grid_func_tmp))

    return grid_func


def inverse_rigid_body_motion(new_coords: ndarray, inv_R: ndarray, b: ndarray, c: ndarray) -> ndarray:
    """
    Based on future coordinates new_coords, find the previous coordinates if rigid body motion occurred.

    Args:
        new_coords: The coordinates at the future time.
        inv_R: The inverse of the rotation matrix.
        b: The vector to the center of rotation.
        c: The translation vector.

    Returns:
        The coordinates at the previous time.
    """
    return np.matmul(inv_R, (new_coords - c)) + b


def gridfunction_rigid_body_motion(t: Parameter, orig_gfu: GridFunction, gfu: GridFunction, inv_R: Callable, mesh: Mesh,
                                   N: List[int], scale: List[float], offset: List[float]) -> GridFunction:
    """
    Construct a gridfunction following rigid body motion of an original field.

    This function currently only handles rigid body rotation as the only application of the diffuse interface method to
    moving domains is currently simulation of impeller motion in a stirred-tank reactor.

    Args:
        t: Parameter containing the current time.
        orig_gfu: Gridfunction containing the initial field.
        gfu: The gridfunction whose values should be updated.
        inv_R: The inverse of the rotation matrix.
        mesh: Structured quadrilateral/hexahedral NGSolve mesh.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).

    Returns:
        Gridfunction containing the field at the current time.
    """

    # TODO: combine these two branches of the if statement together (it's the same code).
    if mesh.dim == 2:
        tmp_arr = np.ones((N[0] + 1, N[1] + 1))

        indices = np.indices((N[0] + 1, N[1] + 1)).transpose().reshape(-1, 2)

        new_coords = -np.array(offset) + np.array(scale) * indices / np.array(N)
        old_coords = np.matmul(inv_R(t.Get()), new_coords.transpose())

        for i in range(len(indices)):
            old_x, old_y = old_coords[:, i]
            ii, jj = indices[i, :]

            if (old_x >= -offset[0]) and (old_x <= scale[0] - offset[0]) and (old_y >= -offset[1]) and \
                    (old_y <= scale[1] - offset[1]):
                # Only update points that are still within the original bounds of the phase field.
                try:
                    mip = mesh(old_x, old_y)
                    tmp_arr[ii, jj] = orig_gfu(mip)
                except:
                    # The bounds of the phase field may extend beyond the bounds of the mesh (ex: a cylindrical mesh
                    # with a rectangular prism phase field). As long as the object being approximated by the phase field
                    # remains within the bounds of the mesh points on the phase field outside said bounds can be
                    # ignored.
                    pass

        gfu.Set(ngs.VoxelCoefficient((-offset[0], -offset[1]), (scale[0] - offset[0], scale[1] - offset[1]),
                                     tmp_arr.transpose(), linear=True))

    elif mesh.dim == 3:
        tmp_arr = np.ones((N[0] + 1, N[1] + 1, N[2] + 1))

        indices = np.indices((N[0] + 1, N[1] + 1, N[2] + 1)).transpose().reshape(-1, 3)

        new_coords = -np.array(offset) + np.array(scale) * indices / np.array(N)
        old_coords = np.matmul(inv_R(t.Get()), new_coords.transpose())

        for i in range(len(indices)):
            old_x, old_y, old_z = old_coords[:, i]
            ii, jj, kk = indices[i, :]

            if (old_x >= -offset[0]) and (old_x <= scale[0] - offset[0]) and (old_y >= -offset[1]) and \
                    (old_y <= scale[1] - offset[1]) and (old_z >= -offset[2]) and (old_z <= scale[2] - offset[2]):
                # Only update points that are still within the original bounds of the phase field.
                try:
                    mip = mesh(old_x, old_y, old_z)
                    tmp_arr[ii, jj, kk] = orig_gfu(mip)
                except:
                    # The bounds of the phase field may extend beyond the bounds of the mesh (ex: a cylindrical mesh
                    # with a rectangular prism phase field). As long as the object being approximated by the phase field
                    # remains within the bounds of the mesh points on the phase field outside said bounds can be
                    # ignored.
                    pass

        gfu.Set(ngs.VoxelCoefficient((-offset[0], -offset[1], -offset[2]), (scale[0] - offset[0], scale[1] - offset[1],
                                                                            scale[2] - offset[2]),
                                     tmp_arr.transpose(2,1,0), linear=True))

    else:
        raise ValueError('Mesh has dimension {}. '
                         '`gridfunction_rigid_body_motion` only works with 2D or 3D meshes.'.format(mesh.dim))

    return gfu
