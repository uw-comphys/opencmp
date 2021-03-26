from config_functions import ConfigParser
from helpers.error import norm
from solvers import Solver
from ngsolve import GridFunction
import tabulate
import math


def h_convergence(config: ConfigParser, solver: Solver, sol: GridFunction) -> None:
    """
    Function to check h (mesh element size) convergence and print results.

    Args:
        config: Config file from which to grab
        solver: The solver used
        sol: Gridfunction that contains the current solution
    """
    num_refinements = config.get_item(['ERROR ANALYSIS', 'num_refinements'], int)
    average_lst = config.get_list(['ERROR ANALYSIS', 'error_average'], str, quiet=True)
    var = config.get_item(['ERROR ANALYSIS', 'convergence_variable'], str)

    component = solver.model.model_components[var]
    average = component in average_lst

    # First solve used the default settings.
    if component is None:
        err = norm('l2_norm', sol, solver.model.ref_sol['ref_sols'][var],
                   solver.model.mesh, solver.model.fes, average)
    else:
        err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var],
                   solver.model.mesh, solver.model.fes.components[component], average)

    # Track the convergence information.
    num_dofs_lst = [solver.model.fes.ndof]
    error_lst = [err]

    # Then run through a series of mesh refinements and resolve on each
    # refined mesh.
    for n in range(num_refinements):
        solver.model.mesh.Refine()
        solver.model.fes.Update()
        solver.reset_model()
        sol = solver.solve()
        if component is None:
            err = norm('l2_norm', sol, solver.model.ref_sol['ref_sols'][var],
                       solver.model.mesh, solver.model.fes, average)
        else:
            err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var],
                       solver.model.mesh, solver.model.fes.components[component], average)

        num_dofs_lst.append(solver.model.fes.ndof)
        error_lst.append(err)

        print('L2 norm at refinement {0}: {1}'.format(n, err))

    # Display the results nicely.
    convergence_table = [['Refinement Level', 'DOFs', 'Error', 'Convergence Rate']]
    convergence_table.append([1, num_dofs_lst[0], error_lst[0], 0])

    for n in range(num_refinements):
        ref_level = '1/{}'.format(int(2 ** (n + 1)))
        convergence_rate = math.log(error_lst[n] / error_lst[n + 1]) / \
                           (math.log(num_dofs_lst[n + 1] / (num_dofs_lst[n] * 2.0 ** (solver.model.mesh.dim - 1))))
        convergence_table.append([ref_level, num_dofs_lst[n + 1], error_lst[n + 1], convergence_rate])

    print(tabulate.tabulate(convergence_table, headers='firstrow', floatfmt=('.1f', '.1f', '.3e', '.2f')))


def p_convergence(config: ConfigParser, solver: Solver, sol: GridFunction) -> None:
    """
    Function to check p (interpolat polynomial order) convergence and print results.

    Args:
        config: Config file from which to grab
        solver: The solver used
        sol: Gridfunction that contains the current solution
    """
    num_refinements = config.get_item(['ERROR ANALYSIS', 'num_refinements'], int)
    average_lst = config.get_list(['ERROR ANALYSIS', 'error_average'], str, quiet=True)
    var = config.get_item(['ERROR ANALYSIS', 'convergence_variable'], str)

    component = solver.model.model_components[var]
    average = component in average_lst

    # First solve used the default settings.
    if component is None:
        err = norm('l2_norm', sol, solver.model.ref_sol['ref_sols'][var],
                   solver.model.mesh, solver.model.fes, average)
    else:
        err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var],
                   solver.model.mesh, solver.model.fes.components[component], average)

    # Track the convergence information.
    num_dofs_lst = [solver.model.fes.ndof]
    interp_ord_lst = [solver.model.interp_ord]
    error_lst = [err]

    # Then run through a series of interpolant refinements.
    for n in range(num_refinements):
        solver.model.interp_ord += 1
        solver.model.fes.Update()
        solver.reset_model()
        sol = solver.solve()
        if component is None:
            err = norm('l2_norm', sol, solver.model.ref_sol['ref_sols'][var],
                       solver.model.mesh, solver.model.fes, average)
        else:
            err = norm('l2_norm', sol.components[component], solver.model.ref_sol['ref_sols'][var],
                       solver.model.mesh, solver.model.fes.components[component], average)

        num_dofs_lst.append(solver.model.fes.ndof)
        interp_ord_lst.append(solver.model.interp_ord)
        error_lst.append(err)

        print('L2 norm at refinement {0}: {1}'.format(n, err))

    # Display the results nicely.
    convergence_table = [['Interpolant Order', 'DOFs', 'Error', 'Convergence Rate']]
    convergence_table.append([interp_ord_lst[0], num_dofs_lst[0], error_lst[0], 0])

    for n in range(num_refinements):
        # TODO: Not sure if this is the correct equation for p-convergence.
        convergence_rate = math.log(error_lst[n] / error_lst[n + 1]) / math.log(num_dofs_lst[n + 1] / num_dofs_lst[n])
        convergence_table.append([interp_ord_lst[n + 1], num_dofs_lst[n + 1], error_lst[n + 1], convergence_rate])

    print(tabulate.tabulate(convergence_table, headers='firstrow', floatfmt=['.1f', '.1f', '.3e', '.2f']))
