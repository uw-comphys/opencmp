"""
Module for time integration schemes.
"""
import ngsolve as ngs
from models import Model
from typing import List
from ngsolve.comp import ProxyFunction
from typing import Tuple


def explicit_euler(model: Model, gfu_0: List[ngs.GridFunction], U: List[ProxyFunction], V: List[ProxyFunction],
                   dt: List[ngs.Parameter]) -> Tuple[ngs.BilinearForm, ngs.LinearForm]:
    """
    Explicit Euler time integration scheme.

    Function to constructs the final bilinear and linear forms for the time integration scheme by adding the
    necessary time-dependent terms to the model's stationary terms.
    The returned bilinear and linear forms have NOT been assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        U: List of trial functions for the model.
        V: List of test (weighting) functions.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        a: The final bilinear form (as a ngs.BilinearForm but not assembled).
        L: The final linear form (as a ngs.LinearForm but not assembled).
    """

    # Separate out the various components of each gridfunction solution to a previous timestep.
    # gfu_lst = [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    gfu_lst = []
    for i in range(len(gfu_0)):
        if (len(gfu_0[i].components) > 0):
            gfu_lst.append([gfu_0[i].components[j] for j in range(len(gfu_0[i].components))])
        else:
            gfu_lst.append([gfu_0[i]])

    # Construct the linear and bilinear forms.
    a = ngs.BilinearForm(model.fes)

    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, gfu_lst[0])
    L += -1.0 * model.construct_bilinear(gfu_lst[0], V, True)

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a += (U[0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
    else:
        a += (U[0] * V[0] / dt[0]) * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * ngs.dx

    return a, L


def implicit_euler(model: Model, gfu_0: List[ngs.GridFunction], U: List[ProxyFunction], V: List[ProxyFunction],
                   dt: List[ngs.Parameter]) -> Tuple[ngs.BilinearForm, ngs.LinearForm]:
    """
    Implicit Euler time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the
    necessary time-dependent terms to the model's stationary terms.
    The returned bilinear and linear forms have NOT been assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        U: List of trial functions for the model.
        V: List of test (weighting) functions.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        a: The final bilinear form (as a ngs.BilinearForm but not assembled).
        L: The final linear form (as a ngs.LinearForm but not assembled).
    """

    # Separate out the various components of each gridfunction solution to a previous timestep.
    # gfu_lst = [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    gfu_lst = []
    for i in range(len(gfu_0)):
        if (len(gfu_0[i].components) > 0):
            gfu_lst.append([gfu_0[i].components[j] for j in range(len(gfu_0[i].components))])
        else:
            gfu_lst.append([gfu_0[i]])

    # Construct the linear and bilinear forms.
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V)

    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, gfu_lst[0])

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a += (U[0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
    else:
        a += (U[0] * V[0] / dt[0]) * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * ngs.dx

    return a, L


def crank_nicolson(model: Model, gfu_0: List[ngs.GridFunction], U: List[ProxyFunction], V: List[ProxyFunction],
                   dt: List[ngs.Parameter]) -> Tuple[ngs.BilinearForm, ngs.LinearForm]:
    """
    Crank Nicolson (trapezoidal rule) time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the
    necessary time-dependent terms to the model's stationary terms.
    The returned bilinear and linear forms have NOT been assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        U: List of trial functions for the model.
        V: List of test (weighting) functions.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        a: The final bilinear form (as a ngs.BilinearForm but not assembled).
        L: The final linear form (as a ngs.LinearForm but not assembled).
    """

    # Separate out the various components of each gridfunction solution to a previous timestep.
    # gfu_lst = [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    gfu_lst = []
    for i in range(len(gfu_0)):
        if (len(gfu_0[i].components) > 0):
            gfu_lst.append([gfu_0[i].components[j] for j in range(len(gfu_0[i].components))])
        else:
            gfu_lst.append([gfu_0[i]])

    # Construct the linear and bilinear forms.
    a = ngs.BilinearForm(model.fes)
    a += 0.5 * model.construct_bilinear(U, V)

    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, gfu_lst[0])
    L += -0.5 * model.construct_bilinear(gfu_lst[0], V, True)

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a += (U[0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
    else:
        a += (U[0] * V[0] / dt[0]) * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * ngs.dx

    return a, L


def CNLF(model: Model, gfu_0: List[ngs.GridFunction], U: List[ProxyFunction], V: List[ProxyFunction],
         dt: List[ngs.Parameter]) -> Tuple[ngs.BilinearForm, ngs.LinearForm]:
    """
    Crank Nicolson Leap Frog IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the
    necessary time-dependent terms to the model's stationary terms.
    The returned bilinear and linear forms have NOT been assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        U: List of trial functions for the model.
        V: List of test (weighting) functions.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        a: The final bilinear form (as a ngs.BilinearForm but not assembled).
        L: The final linear form (as a ngs.LinearForm but not assembled).
    """

    # Separate out the various components of each gridfunction solution to a previous timestep.
    # gfu_lst = [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    gfu_lst = []
    for i in range(len(gfu_0)):
        if (len(gfu_0[i].components) > 0):
            gfu_lst.append([gfu_0[i].components[j] for j in range(len(gfu_0[i].components))])
        else:
            gfu_lst.append([gfu_0[i]])

    # Construct the linear and bilinear forms.
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V)

    L = ngs.LinearForm(model.fes)
    L += 2.0 * model.construct_linear(V, gfu_lst[0])
    L += -1.0 * model.construct_bilinear(gfu_lst[1], V, True)

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a += (U[0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
        L += (gfu_lst[1][0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
    else:
        a += (U[0] * V[0] / dt[0]) * ngs.dx
        L += (gfu_lst[1][0] * V[0] / dt[0]) * ngs.dx

    return a, L


def SBDF(model: Model, gfu_0: List[ngs.GridFunction], U: List[ProxyFunction], V: List[ProxyFunction],
         dt: List[ngs.Parameter]) -> Tuple[ngs.BilinearForm, ngs.LinearForm]:
    """
    Third order semi-implicit backwards differencing time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the
    necessary time-dependent terms to the model's stationary terms.
    The returned bilinear and linear forms have NOT been assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        U: List of trial functions for the model.
        V: List of test (weighting) functions.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        a: The final bilinear form (as a ngs.BilinearForm but not assembled).
        L: The final linear form (as a ngs.LinearForm but not assembled).
    """

    # Separate out the various components of each gridfunction solution to a previous timestep.
    # gfu_lst = [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    gfu_lst = []
    for i in range(len(gfu_0)):
        if (len(gfu_0[i].components) > 0):
            gfu_lst.append([gfu_0[i].components[j] for j in range(len(gfu_0[i].components))])
        else:
            gfu_lst.append([gfu_0[i]])

    # Construct the linear and bilinear forms.
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V)

    L = ngs.LinearForm(model.fes)
    L += 3.0 * model.construct_linear(V, gfu_lst[0])
    L += -3.0 * model.construct_linear(V, gfu_lst[1])
    L += model.construct_linear(V, gfu_lst[2])

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a += (11.0 / 6.0) * (U[0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
        L += (3.0 * gfu_lst[0][0] - 1.5 * gfu_lst[1][0] + gfu_lst[2][0] / 3.0) * V[0] / dt[0] * model.DIM_solver.phi_gfu * ngs.dx
    else:
        a += (11.0 / 6.0) * (U[0] * V[0] / dt[0]) * ngs.dx
        L += (3.0 * gfu_lst[0][0] - 1.5 * gfu_lst[1][0] + gfu_lst[2][0] / 3.0) * V[0] / dt[0] * ngs.dx

    return a, L

def adaptive_IMEX_pred(model: Model, gfu_0: List[ngs.GridFunction], U: List[ProxyFunction], V: List[ProxyFunction],
         dt: List[ngs.Parameter]) -> Tuple[ngs.BilinearForm, ngs.LinearForm]:
    """
    Predictor for the adaptive time-stepping IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the
    necessary time-dependent terms to the model's stationary terms.
    The returned bilinear and linear forms have NOT been assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        U: List of trial functions for the model.
        V: List of test (weighting) functions.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        a: The final bilinear form (as a ngs.BilinearForm but not assembled).
        L: The final linear form (as a ngs.LinearForm but not assembled).
    """

    # Separate out the various components of each gridfunction solution to a previous timestep.
    # gfu_lst = [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    gfu_lst = []
    for i in range(len(gfu_0)):
        if (len(gfu_0[i].components) > 0):
            gfu_lst.append([gfu_0[i].components[j] for j in range(len(gfu_0[i].components))])
        else:
            gfu_lst.append([gfu_0[i]])

    # Operators specific to this time integration scheme.
    w = dt[0] / dt[1]
    E = model.construct_gfu().components[0]
    E_expr = (1.0 + w) * gfu_lst[0][0] - w * gfu_lst[1][0]
    E.Set(E_expr)

    # Construct the linear and bilinear forms.
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V)

    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, [gfu_lst[0][0], 0.0]) # TODO: Why doesn't using E work? Numerical error?

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a += (U[0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * model.DIM_solver.phi_gfu * ngs.dx
    else:
        a += (U[0] * V[0] / dt[0]) * ngs.dx
        L += (gfu_lst[0][0] * V[0] / dt[0]) * ngs.dx

    return a, L