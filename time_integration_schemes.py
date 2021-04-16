"""
Copyright 2021 the authors (see AUTHORS file for full list)

This file is part of OpenCMP.

OpenCMP is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

OpenCMP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with OpenCMP.  If not, see <https://www.gnu.org/licenses/>.
"""

import ngsolve as ngs
from models import Model
from typing import List
from ngsolve.comp import ProxyFunction
from ngsolve import dx
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

    # Construct the bilinear form
    a = ngs.BilinearForm(model.fes)

    # Construct the linear form
    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, gfu_lst[0], dt[0])
    L += -1.0 * model.construct_bilinear(gfu_lst[0], V, dt[0], True)

    a_dt, L_dt = model.time_derivative_terms(gfu_lst, 'explicit euler')

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a_dt *= model.DIM_solver.phi_gfu
        L_dt *= model.DIM_solver.phi_gfu

    a += a_dt * dx
    L += L_dt * dx

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

    # Construct the bilinear form
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V, dt[0])

    # Construct the linear form
    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, gfu_lst[0], dt[0])

    a_dt, L_dt = model.time_derivative_terms(gfu_lst, 'implicit euler')

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a_dt *= model.DIM_solver.phi_gfu
        L_dt *= model.DIM_solver.phi_gfu

    a += a_dt * dx
    L += L_dt * dx

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

    # Construct the bilinear form
    a = ngs.BilinearForm(model.fes)
    a += 0.5 * model.construct_bilinear(U, V, dt[0])

    # Construct the linear form
    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, gfu_lst[0], dt[0])
    L += -0.5 * model.construct_bilinear(gfu_lst[0], V, dt[0], True)

    a_dt, L_dt = model.time_derivative_terms(gfu_lst, 'crank nicolson')

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a_dt *= model.DIM_solver.phi_gfu
        L_dt *= model.DIM_solver.phi_gfu

    a += a_dt * dx
    L += L_dt * dx

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

    # Construct the bilinear form
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V, dt[0])

    # Construct the linear form
    L = ngs.LinearForm(model.fes)
    L += 2.0 * model.construct_linear(V, gfu_lst[0], dt[0])
    L += -1.0 * model.construct_bilinear(gfu_lst[1], V, dt[0], True)

    a_dt, L_dt = model.time_derivative_terms(gfu_lst, 'CNLF')

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a_dt *= model.DIM_solver.phi_gfu
        L_dt *= model.DIM_solver.phi_gfu

    a += a_dt * dx
    L += L_dt * dx

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

    # Construct the bilinear form
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V, dt[0])

    # Construct the linear form
    L = ngs.LinearForm(model.fes)
    L += 3.0 * model.construct_linear(V, gfu_lst[0], dt[0])
    L += -3.0 * model.construct_linear(V, gfu_lst[1], dt[0])
    L += model.construct_linear(V, gfu_lst[2], dt[0])

    a_dt, L_dt = model.time_derivative_terms(gfu_lst, 'SBDF')

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a_dt *= model.DIM_solver.phi_gfu
        L_dt *= model.DIM_solver.phi_gfu

    a += a_dt * dx
    L += L_dt * dx

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

    # Construct the bilinear form
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear(U, V, dt[0])

    # Construct the linear form
    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, [gfu_lst[0][0], 0.0], dt[0])  # TODO: Why doesn't using E work? Numerical error?

    a_dt, L_dt = model.time_derivative_terms(gfu_lst, 'crank nicolson')

    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        a_dt *= model.DIM_solver.phi_gfu
        L_dt *= model.DIM_solver.phi_gfu

    a += a_dt * dx
    L += L_dt * dx

    return a, L
