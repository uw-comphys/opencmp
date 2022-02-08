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

from opencmp.config_functions import ConfigParser
from .error import norm
from opencmp.solvers import Solver
from ngsolve import GridFunction
import math
from typing import List, Union
from .misc import can_import_module

if can_import_module('tabulate'):
    missing_tabulate = False
    import tabulate
else:
    missing_tabulate = True


def h_convergence(config: ConfigParser, solver: Solver, sol: GridFunction, var: str) -> None:
    """
    Function to check h (mesh element size) convergence and print results.

    Args:
        config: Config file from which to grab.
        solver: The solver used.
        sol: Gridfunction that contains the current solution.
        var: The variable of interest.
    """
    if missing_tabulate:
        raise ImportError('tabulate module is not installed. Install it with `pip install tabulate`.')

    num_refinements = config.get_item(['ERROR ANALYSIS', 'num_refinements'], int)
    average_lst = config.get_list(['ERROR ANALYSIS', 'error_average'], str, quiet=True)
    component = solver.model.model_components[var]
    average = component in average_lst

    # First solve used the default settings.
    # NOTE: Assuming the t^n+1 value of the reference solution should always be used.
    err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var][0],
                   solver.model.mesh, solver.model.fes.components[component], average)

    # Track the convergence information.
    num_dofs_lst = [solver.model.mesh.ne]
    error_lst = [err]

    # Then run through a series of mesh refinements and resolve on each
    # refined mesh.
    for n in range(num_refinements):
        solver.model.mesh.Refine()
        solver.model.fes.Update()
        solver.reset_model()
        sol = solver.solve()

        # NOTE: Assuming the t^n+1 value of the reference solution should always be used.
        err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var][0],
                       solver.model.mesh, solver.model.fes.components[component], average)

        num_dofs_lst.append(solver.model.mesh.ne)
        error_lst.append(err)

        print('L2 norm at refinement {0}: {1}'.format(n+1, err))

    # Display the results nicely.
    convergence_table: List[List[Union[str, float, int]]] = [['Refinement Level', 'Mesh Elements', 'Error', 'Convergence Rate']]
    convergence_table.append([1, num_dofs_lst[0], error_lst[0], 0])

    for n in range(num_refinements):
        ref_level = '1/{}'.format(int(2 ** (n + 1)))
        convergence_rate = math.log(error_lst[n] / error_lst[n + 1]) / \
                           (math.log(num_dofs_lst[n + 1] / (num_dofs_lst[n] * 2.0 ** (solver.model.mesh.dim - 1))))
        convergence_table.append([ref_level, num_dofs_lst[n + 1], error_lst[n + 1], convergence_rate])

    print(tabulate.tabulate(convergence_table, headers='firstrow', floatfmt=('.1f', '.1f', '.3e', '.2f')))


def p_convergence(config: ConfigParser, solver: Solver, sol: GridFunction, var: str) -> None:
    """
    Function to check p (interpolat polynomial order) convergence and print results.

    Args:
        config: Config file from which to grab.
        solver: The solver used.
        sol: Gridfunction that contains the current solution.
        var: The variable of interest.
    """
    num_refinements = config.get_item(['ERROR ANALYSIS', 'num_refinements'], int)
    average_lst = config.get_list(['ERROR ANALYSIS', 'error_average'], str, quiet=True)
    component = solver.model.model_components[var]
    average = component in average_lst

    # First solve used the default settings.
    # NOTE: Assuming the t^n+1 value of the reference solution should always be used.
    err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var][0],
                   solver.model.mesh, solver.model.fes.components[component], average)

    # Track the convergence information.
    num_dofs_lst = [solver.model.fes.ndof]
    interp_ord_lst = [solver.model.interp_ord]
    error_lst = [err]

    # Then run through a series of interpolant refinements.
    for n in range(num_refinements):
        solver.model.interp_ord += 1
        solver.model.load_mesh_fes(mesh=False, fes=True)
        solver.reset_model()
        sol = solver.solve()

        # NOTE: Assuming the t^n+1 value of the reference solution should always be used.
        err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var][0],
                       solver.model.mesh, solver.model.fes.components[component], average)

        num_dofs_lst.append(solver.model.fes.ndof)
        interp_ord_lst.append(solver.model.interp_ord)
        error_lst.append(err)

        print('L2 norm at refinement {0}: {1}'.format(n+1, err))

    # Display the results nicely.
    convergence_table: List[List[Union[str, float, int]]] = [['Interpolant Order', 'DOFs', 'Error', 'Convergence Rate']]
    convergence_table.append([interp_ord_lst[0], num_dofs_lst[0], error_lst[0], 0])

    for n in range(num_refinements):
        # TODO: Not sure if this is the correct equation for p-convergence.
        convergence_rate = math.log(error_lst[n] / error_lst[n + 1]) / math.log(num_dofs_lst[n + 1] / num_dofs_lst[n])
        convergence_table.append([interp_ord_lst[n + 1], num_dofs_lst[n + 1], error_lst[n + 1], convergence_rate])

    print(tabulate.tabulate(convergence_table, headers='firstrow', floatfmt=['.1f', '.1f', '.3e', '.2f']))
