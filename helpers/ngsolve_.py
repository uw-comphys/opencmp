import ngsolve as ngs
from typing import List, Tuple, Union
from ngsolve.comp import ProxyFunction
import numpy as np


def construct_p_mat(p: Union[float, ProxyFunction], dim: int) -> ngs.CoefficientFunction:
    """
    Constructs the pressure identity matrix (pI) used by Stokes and
    incompressible Navier-Stokes flow in boundary integrals.

    Args:
        p: The pressure at the boundary.
        dim: The dimension of the matrix.

    Returns:
        ~: The pI matrix at the boundary.
    """

    if (dim == 2):
        return ngs.CoefficientFunction((p, 0.0, 0.0, p), dims=(2, 2))
    elif (dim == 3):
        return ngs.CoefficientFunction((p, 0.0, 0.0, 0.0, p, 0.0, 0.0, 0.0, p), dims=(3, 3))
    else:
        print('Only supports 2D and 3D meshes, not {}D meshes.'.format(dim))


def get_special_functions(mesh: ngs.Mesh, nu: float) \
        -> Tuple[ngs.CoefficientFunction, ngs.CoefficientFunction, ngs.CoefficientFunction]:
    """
    Generates a set of special functions needed for DG so they don't need to be 
    rewritten multiple times.
    
    Args:
        mesh: The mesh used for the simulation.
        nu: The penalty parameter for interior penalty method DG.
        
    Returns:
        n: The unit normal for every facet of the mesh.
        h: The "size" of every mesh element.
        alpha: The penalty coefficient.
    """

    n = ngs.specialcf.normal(mesh.dim)
    h = ngs.specialcf.mesh_size
    alpha = nu / h

    return n, h, alpha


def NGSolve_to_numpy(mesh: ngs.Mesh, gfu: ngs.GridFunction, N: List, scale: List, offset: List, dim: int = 2,
                     binary: Union[None, np.ndarray] = None) -> np.ndarray:
    """
    Assign the values in gfu to a numpy array whose elements correspond to the
    nodes of mesh while preserving spatial ordering. The output array only
    contains node values, never the values of higher order dofs.

    Args:
        mesh (NGSolve mesh): Structured quadrilateral/hexahedral NGSolve mesh.
        gfu (NGSolve GridFunction): NGSolve GridFunction containing the values
                                    to assign.
        N (list): Number of mesh elements in each direction (N+1 nodes).
        scale (list): Extent of the meshed domain in each direction ([-2,2]
                      square -> scale=[4,4]).
        offset (list): Centers the meshed domain in each direction ([-2,2]
                       square -> offset=[2,2]).
        dim (int): Dimension of the domain (must be 2 or 3).
        binary (None or numpy array): If exists, arr is multiplied by binary to zero out any
                                      array elements outside of the approximated complex
                                      geometry.

    Returns:
        arr_vec (list): List of arrays containing the values in gfu. Since gfu may be vector-valued, the arrays in
                        arr_vec correspond to the various components of gfu.
    """

    if dim == 2:
        arr = np.zeros((N[0] + 1, N[1] + 1))

        # The GridFunction may be vector-valued.
        # Return a list containing the spatial field for each
        # vector component.
        arr_vec = [arr] * gfu.dim

        for i in range(N[0] + 1):
            for j in range(N[1] + 1):
                x = -offset[0] + scale[0] * i / N[0]
                y = -offset[1] + scale[1] * j / N[1]
                z = 0

                mip = mesh(x, y)
                val = gfu(mip)

                if isinstance(val, tuple):
                    for k in range(len(val)):
                        arr_vec[k][i, j] = val[k]
                else:
                    arr_vec[0][i, j] = val

    elif dim == 3:
        arr = np.zeros((N[0] + 1, N[1] + 1, N[2] + 1))

        # The GridFunction may be vector-valued.
        # Return a list containing the spatial field for each
        # vector component.
        arr_vec = [arr] * gfu.dim

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
        raise ValueError('Only works with 2D or 3D meshes.')

    if binary is not None:
        for arr in arr_vec:
            arr *= binary

    return arr_vec


def numpy_to_NGSolve(mesh: ngs.Mesh, interp_ord: int, arr: np.ndarray, scale: List, offset: List, dim: int = 2) \
        -> ngs.GridFunction:
    """
    Assign the values in arr to a NGSolve GridFunction defined over the given
    NGSolve mesh while preserving spatial ordering.

    Args:
        mesh (NGSolve mesh): Mesh used by the problem the GridFunction will be needed to solve.
        interp_ord (int): Interpolant order of the finite element space for the problem.
        arr (numpy array): Array containing the values to assign.
        scale (list): Extent of the meshed domain in each direction ([-2,2]
                      square -> scale=[4,4]).
        offset (list): Centers the meshed domain in each direction ([-2,2]
                       square -> offset=[2,2]).
        dim (int): Dimension of the domain (must be 2 or 3).

    Returns:
        grid_func (NGSolve GridFunction): Contains the values in arr.
    """

    if dim == 2:
        arr = np.transpose(arr)  # Corrects for x,y mismatch.
        func = ngs.VoxelCoefficient((-offset[0], -offset[1]), (scale[0] - offset[0], scale[1] - offset[1]), arr,
                                    linear=True)
    elif dim == 3:
        arr = np.transpose(arr, (2, 1, 0))  # Corrects for x,y,z mismatch.
        func = ngs.VoxelCoefficient((-offset[0], -offset[1], -offset[2]),
                                    (scale[0] - offset[0], scale[1] - offset[1], scale[2] - offset[2]), arr,
                                    linear=True)
    else:
        raise ValueError('Only works with 2D or 3D meshes.')

    fes = ngs.H1(mesh, order=interp_ord)
    grid_func_tmp = ngs.GridFunction(fes)
    grid_func_tmp.Set(func)

    # Smooths out irregularities caused by the interpolation of the VoxelCoefficient onto the finite element space.
    grid_func = ngs.GridFunction(fes)
    grid_func.Set(ngs.Norm(grid_func_tmp))

    return grid_func


def inverse_rigid_body_motion(new_coords: np.ndarray, inv_R: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Based on future coordinates new_coords, find the previous coordinates if rigid body motion occurred.

    Args:
        new_coords: The coordinates at the future time.
        inv_R: The inverse of the rotation matrix.
        b: The vector to the center of rotation.
        c: The translation vector.

    Returns:
        ~ : The coordinates at the previous time.
    """
    return np.matmul(inv_R, (new_coords - c)) + b


def gridfunction_rigid_body_motion(mesh: ngs.Mesh, old_gfu: ngs.GridFunction, fes: ngs.FESpace, inv_R: np.ndarray,
                                   b: np.ndarray, c: np.ndarray, N: List, scale: List, offset: List,
                                   dim: int) -> ngs.GridFunction:
    """
    Construct a gridfunction following rigid body motion of an original field.

    Args:
        mesh: Structured quadrilateral/hexahedral NGSolve mesh.
        old_gfu: Gridfunction containing the field at the previous time step.
        fes: NGSolve FEM space used by the problem the gridfunction will be needed to solve.
        inv_R: The inverse of the rotation matrix at the previous time.
        b: The vector to the center of rotation.
        c: The translation vector at the previous time.
        N: Number of mesh elements in each direction (N+1 nodes).
        scale: Extent of the meshed domain in each direction ([-2,2] square -> scale=[4,4]).
        offset: Centers the meshed domain in each direction ([-2,2] square -> offset=[2,2]).
        dim: Dimension of the domain (must be 2 or 3).

    Returns:
        gfu: Gridfunction containing the field at the future time.
    """

    if dim == 2:
        tmp_arr = np.zeros((N[0] + 1, N[1] + 1))

        for i in range(N[0] + 1):
            for j in range(N[1] + 1):
                x = -offset[0] + scale[0] * i / N[0]
                y = -offset[1] + scale[1] * j / N[1]

                old_x, old_y = inverse_rigid_body_motion(np.array([x, y]).transpose(), inv_R, b, c)

                if (old_x >= -offset[0]) and (old_x <= scale[0] - offset[0]) and (old_y >= -offset[1]) \
                        and (old_y <= scale[1] - offset[1]):
                    mip = mesh(old_x, old_y)
                    tmp_arr[i, j] = old_gfu(mip)

    elif dim == 3:
        tmp_arr = np.zeros((N[0] + 1, N[1] + 1, N[2] + 1))

        for i in range(N[0] + 1):
            for j in range(N[1] + 1):
                for k in range(N[2] + 1):
                    x = -offset[0] + scale[0] * i / N[0]
                    y = -offset[1] + scale[1] * j / N[1]
                    z = -offset[2] + scale[2] * k / N[2]

                    old_x, old_y, old_z = inverse_rigid_body_motion(np.array([x, y, z]).transpose(), inv_R, b, c)

                    if (old_x >= -offset[0]) and (old_x <= scale[0] - offset[0]) and (old_y >= -offset[1]) \
                            and (old_y <= scale[1] - offset[1]) and (old_z >= -offset[2]) and (old_z <= scale[2] -
                                                                                               offset[2]):
                        mip = mesh(old_x, old_y, old_z)
                        tmp_arr[i, j, j] = old_gfu(mip)

    else:
        raise ValueError('Only works with 2D or 3D meshes.')

    gfu = numpy_to_NGSolve(fes, tmp_arr, scale, offset, dim)

    return gfu
