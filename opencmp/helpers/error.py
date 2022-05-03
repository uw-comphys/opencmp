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

from typing import Union, Optional, List, TYPE_CHECKING, Any

# Sphinx runs into a circular import with `from models import Model`, so only
# import Model for type checking.
if TYPE_CHECKING:
    from ..models import Model
else:
    Model = None

import ngsolve as ngs
import numpy as np
from ngsolve import CoefficientFunction, FESpace, GridFunction, Mesh

from ..config_functions import ConfigParser
from ..helpers.ngsolve_ import get_special_functions


def mean_to_zero(gfu: GridFunction, fes: FESpace, mesh: Mesh) -> GridFunction:
    """
    Function to bias a gridfunction so its mean is zero.

    Args:
        gfu: The gridfunction to modify (or a component of a gridfunction).
        fes: The finite element space the gridfunction is defined on.
        mesh: The mesh the gridfunction is defined on.

    Returns:
        The modified gridfunction.
    """

    avg = mean(gfu, mesh)
    gfu_biased = ngs.GridFunction(fes)
    gfu_biased.Set(gfu - avg)

    return gfu_biased

def mean(gfu: Union[GridFunction, CoefficientFunction], mesh: Mesh) -> float:
    """
    Function to calculate the mean value of a gridfunction.

    Args:
        gfu: The gridfunction of interest.
        mesh: The mesh the gridfunction is defined on.

    Returns:
        The mean value of the gridfunction.
    """

    avg = ngs.sqrt(ngs.Integrate(gfu ** 2, mesh))

    return avg

def _l_1(sol: GridFunction, ref_sol: Union[GridFunction, CoefficientFunction], mesh: Mesh) -> float:
    """ L1 norm """
    return ngs.Integrate(ngs.sqrt((sol - ref_sol)*(sol - ref_sol)), mesh)


def _l_2(sol: GridFunction, ref_sol: Union[GridFunction, CoefficientFunction], mesh: Mesh) -> float:
    """ L2 norm """
    return ngs.sqrt(ngs.Integrate((sol - ref_sol) * (sol - ref_sol), mesh))


def _l_inf(sol: GridFunction, ref_sol: Union[GridFunction, CoefficientFunction], mesh: Mesh, fes: FESpace) -> float:
    """ L-infinity norm """
    gfu_tmp = ngs.GridFunction(fes)
    gfu_tmp.Set(sol - ref_sol)

    # NGSolve has no builtin method for evaluating the L-infinity norm and recommends using numpy.
    arr = gfu_tmp.vec.FV().NumPy()

    return np.max(np.abs(arr))


def norm(norm_type: str, sol: GridFunction, ref_sol: Union[GridFunction, CoefficientFunction],
         mesh: Mesh, fes: FESpace, average: bool) -> float:
    """
    Function to calculate some norm between a solution and an "exact" solution.

    Uses NGSolve's Integrate function. The "exact" solution can be a known exact solution or a reference solution
    previously loaded from file.

    Args:
        norm_type: Which norm to calculate.
        sol: The solution GridFunction.
        ref_sol: The "exact" solution to compare against.
        mesh: The mesh to compare the solutions on.
        fes: The finite element space that the solutions come from (or a component of it).
        average: If true offset each solution by its average (use for variables like pressure that are only
            solved up to a constant).

    Returns:
        The calculated error using the specified norm.
    """

    if average:
        ref_sol_tmp = mean_to_zero(ref_sol, fes, mesh)
        sol_tmp = mean_to_zero(sol, fes, mesh)
    else:
        ref_sol_tmp = ref_sol
        sol_tmp = sol

    if norm_type == 'l1_norm':
        err = _l_1(sol_tmp, ref_sol_tmp, mesh)
    elif norm_type == 'l2_norm':
        err = _l_2(sol_tmp, ref_sol_tmp, mesh)
    elif norm_type == 'linfinity_norm':
        err = _l_inf(sol_tmp, ref_sol_tmp, mesh, fes)
    else:
        raise ValueError('{} has not been implemented yet.'.format(norm_type))

    return err


def _facet_jumps(sol: GridFunction, mesh: Mesh) -> float:
    """
    Function to check how continuous the solution is across mesh facets.

    This is mainly of interest when DG is used. Continuous Galerkin FEM solutions will always be perfectly continuous
    across facets.

    Args:
        sol: The solution GridFunction.
        mesh: The mesh that was solved on.

    Returns:
        The L2 norm of the facet jumps.
    """

    mag_jumps = ngs.sqrt(ngs.Integrate((sol - sol.Other())**2 * ngs.dx(element_boundary=True), mesh))

    return mag_jumps


def _divergence(sol: GridFunction, mesh: Mesh) -> float:
    """
    Function to calculate the divergence of a variable over the domain.

    Use with velocity to confirm that conservation of mass is being satisfied.

    Args:
        sol: The solution GridFunction.
        mesh: The mesh that was solved on.

    Returns:
        The L2 norm of the divergence of the field.
    """

    div_var = ngs.sqrt(ngs.Integrate((ngs.div(sol)**2), mesh))

    return div_var


def _surface_traction(sol: GridFunction, model: Model, marker: Optional[str]) -> CoefficientFunction:
    """
    Function to calculate the surface traction on an object described by a given mesh surface.

    Args:
        sol: The solution gridfunction.
        model: The solved model to calculate the error for.
        marker: The mesh marker for the surface of the object. If the diffuse interface method is being used this is not
            needed since the phase field will be used to locate the object.

    Returns:
        The surface traction on the object due to the flow field.
    """

    if not 'u' in model.model_components.keys() and 'p' in model.model_components.keys():
        raise ValueError('Surface traction can only be computed for models with fluid flow.')
    else:
        u_comp = model.model_components['u']
        p_comp = model.model_components['p']

    # Need some of the special ngsolve helper functions.
    n, h, alpha, I_mat = get_special_functions(model.mesh, model.nu)

    # Also need the kinematic viscosity at the current time step. Any model that involves flow should have a kinematic
    # viscosity. Otherwise throw an error, surface traction is nonsensical for the model.
    assert hasattr(model, 'kv')
    kv = model.kv[0]

    # If using L2 spaces for pressure need to use a special type of coefficient function to be able to evaluate boundary
    # integrals.
    if model.element['p'] == 'L2':
        p_mat = ngs.BoundaryFromVolumeCF(sol.components[p_comp]) * I_mat
    else:
        p_mat = sol.components[p_comp] * I_mat

    grad_u = ngs.Grad(sol.components[u_comp])

    if isinstance(marker, str):
        # A conformal mesh is being used.
        #
        # Confirm that the marker name corresponds to an actual mesh surface.

        if not (marker in model.mesh.GetBoundaries()):
            print(type(marker), type(model.mesh.GetBoundaries()), type(model.mesh.GetBoundaries()[0]))
            raise ValueError('{} is not a mesh surface.'.format(marker))

        # Note that the rho*u*u component of the stress tensor is not included because it is assumed that there is no
        # mass transfer through the surface.
        surface_traction = ngs.Integrate(cf=(kv * grad_u.trans * n + kv * grad_u * n - p_mat * n), mesh=model.mesh, definedon=model.mesh.Boundaries(marker))

    elif marker is None:
        # The diffuse interface method is being used.
        #
        # Confirm that DIM is actually being used.
        if not model.DIM:
            raise ValueError('The diffuse interface method is not being used, so a mesh surface must be given to locate'
                             ' the surface of the object.')

        # Note that the rho*u*u component of the stress tensor is not included because it is assumed that there is no
        # mass transfer through the surface.
        # Also note that phi is assumed to be 1 in the fluid and 0 in the object, so a (1-phi) term is added to ensure
        # that the surface traction integral only occurs over the outer surface of the object. Otherwise the surface
        # traction integral would double count the inner and outer surfaces of the object.
        surface_traction = ngs.Integrate((kv * grad_u - p_mat) * -model.DIM_solver.grad_phi_gfu, model.mesh)

    return surface_traction


def calc_error(config: ConfigParser, model: Model, sol: GridFunction) -> List:
    """
    Function to calculate L2 error and other error metrics and print them.

    Args:
        config: Config file from which to grab.
        model: The solved model to calculate the error for.
        sol: Gridfunction that contains the current solution.

    Returns:
        A list of all of the calculated error metrics ordered by the order of the model.ref_sol['metrics'] dictionary.
    """

    average_lst = config.get_list(['ERROR ANALYSIS', 'error_average'], str, quiet=True)
    norm_lst = ['l1_norm', 'l2_norm', 'linfinity_norm']

    error_lst: List[Any] = []

    if model.ref_sol['metrics']:
        for metric, var_lst in model.ref_sol['metrics'].items():
            if metric.lower() in norm_lst:
                # Calculate norms.
                for var in var_lst:
                    ref_sol = model.ref_sol['ref_sols'][var][0] # Assuming the t^n+1 value of the reference solution should always be used.
                    average = var in average_lst
                    component = model.model_components[var]
                    err = norm(metric.lower(), sol.components[component], ref_sol, model.mesh, model.fes.components[component], average)

                    error_lst.append(err)
                    print('{0} in {1}: {2}'.format(metric.replace('_', ' '), var, err))

            elif metric == 'divergence':
                # Calculate divergence.
                for var in var_lst:
                    component = model.model_components[var]
                    div_var = _divergence(sol.components[component], model.mesh)

                    error_lst.append(div_var)
                    print('divergence of {0}: {1}'.format(var, div_var))

            elif metric == 'facet_jumps':
                # Calculate facet jumps.
                for var in var_lst:
                    component = model.model_components[var]
                    mag_jumps = _facet_jumps(sol.components[component], model.mesh)

                    error_lst.append(mag_jumps)
                    print('magnitude of jump of {0} facets: {1}'.format(var, mag_jumps))

            elif metric == 'surface_traction':
                # Calculate the surface traction.
                for marker in var_lst:
                    if marker == 'None':
                        surface_traction = _surface_traction(sol, model, None)
                        error_lst.append([st for st in surface_traction])
                        print('surface traction: {}'.format([st for st in surface_traction]))
                    else:
                        surface_traction = _surface_traction(sol, model, marker)
                        error_lst.append([st for st in surface_traction])
                        print('surface traction on {0}: {1}'.format(marker, [st for st in surface_traction]))
            else:
                raise ValueError('{} has not been implemented yet.'.format(metric))

    return error_lst
