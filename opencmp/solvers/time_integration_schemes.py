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
from ngsolve import GridFunction, BilinearForm, LinearForm, Parameter
from opencmp.models import Model
from typing import List, Tuple
from ngsolve import dx


def explicit_euler(model: Model, gfu_0: List[GridFunction], dt: List[Parameter])\
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Explicit Euler time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # All the terms use the same dt.
    tmp_dt = dt[0]

    # Construct the bilinear forms
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        a_tmp = BilinearForm(model.fes)

        a_tmp += a_lst[i]

        a.append(a_tmp)

    # Construct the linear forms
    L : List[BilinearForm]  = []
    L_ode                   = model.construct_bilinear_time_ODE(gfu_lst[1], V, tmp_dt, 1)
    L_lst                   = model.construct_linear(V, gfu_lst[1], tmp_dt, 1)
    for i in range(model.num_weak_forms):
        L_tmp = LinearForm(model.fes)

        L_tmp += -1.0 * L_ode[i]
        L_tmp += L_lst[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'explicit euler')

    return a, L


def implicit_euler(model: Model, gfu_0: List[GridFunction], dt: List[Parameter], step: int = 0) \
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Implicit Euler time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.
        step: Used for adaptive_three_step to ensure the correct boundary condition and model function values are used
            when the half steps are taken.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # All the terms use the same dt.
    tmp_dt = dt[step]

    # Construct the bilinear form
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, step)
    a_ode                   = model.construct_bilinear_time_ODE(U, V, tmp_dt, step)
    for i in range(model.num_weak_forms):
        a_tmp = BilinearForm(model.fes)

        a_tmp += a_lst[i]
        a_tmp += a_ode[i]

        a.append(a_tmp)

    # Construct the linear form
    L : List[BilinearForm]  = []
    L_lst                   = model.construct_linear(V, None, tmp_dt, step)
    for i in range(model.num_weak_forms):
        L_tmp = LinearForm(model.fes)

        L_tmp += L_lst[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'implicit euler')

    return a, L


def crank_nicolson(model: Model, gfu_0: List[GridFunction], dt: List[Parameter])\
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Crank Nicolson (trapezoidal rule) time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # All the terms use the same dt.
    tmp_dt = dt[0]

    # Construct the bilinear form
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
    a_ode                   = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        a_tmp = ngs.BilinearForm(model.fes)

        a_tmp += a_lst[i]
        a_tmp += 0.5 * a_ode[i]

        a.append(a_tmp)

    # Construct the linear form
    L: List[BilinearForm]   = []
    L_ode                   = model.construct_bilinear_time_ODE(gfu_lst[1], V, tmp_dt, 1)
    L_lst_0                 = model.construct_linear(V, gfu_lst[0], tmp_dt, 0)
    L_lst_1                 = model.construct_linear(V, gfu_lst[1], tmp_dt, 1)
    for i in range(model.num_weak_forms):
        L_tmp = LinearForm(model.fes)

        L_tmp += -0.5 * L_ode[i]
        L_tmp += 0.5 * L_lst_0[i]
        L_tmp += 0.5 * L_lst_1[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'crank nicolson')

    return a, L


def euler_IMEX(model: Model, gfu_0: List[GridFunction], dt: List[Parameter])\
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    First order IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # All the terms use the same dt.
    tmp_dt = dt[0]

    # Construct the bilinear form
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
    a_ode                   = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        a_tmp = BilinearForm(model.fes)

        a_tmp += a_lst[i]
        a_tmp += a_ode[i]

        a.append(a_tmp)

    # Construct the linear form
    L : List[BilinearForm]  = []
    L_lst                   = model.construct_linear(V, None, tmp_dt, 0)
    L_imex                  = model.construct_imex_explicit(V, gfu_lst[1], tmp_dt, 0)
    for i in range(model.num_weak_forms):
        L_tmp = ngs.LinearForm(model.fes)

        L_tmp += L_lst[i]
        L_tmp += L_imex[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'implicit euler')

    return a, L


def CNLF(model: Model, gfu_0: List[GridFunction], dt: List[Parameter]) \
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Crank Nicolson Leap Frog IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # All the terms use the same dt.
    tmp_dt = dt[0]

    # Construct the bilinear forms
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
    a_ode                   = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        a_tmp = BilinearForm(model.fes)

        a_tmp += 2.0 * a_lst[i]
        a_tmp += a_ode[i]

        a.append(a_tmp)

    # Construct the linear form
    L : List[BilinearForm]  = []
    L_lst                   = model.construct_linear(V, gfu_lst[2], tmp_dt, 2)
    L_ode                   = model.construct_bilinear_time_ODE(gfu_lst[2], V, tmp_dt, 2)
    L_imex                  = model.construct_imex_explicit(V, gfu_lst[1], tmp_dt, 1)
    for i in range(model.num_weak_forms):
        L_tmp = LinearForm(model.fes)

        L_tmp += L_lst[i]
        L_tmp += -1.0 * L_ode[i]
        L_tmp += 2.0 * L_imex[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'CNLF')

    return a, L


def SBDF(model: Model, gfu_0: List[GridFunction], dt: List[Parameter]) \
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Third order semi-implicit backwards differencing time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # All the terms use the same dt.
    tmp_dt = dt[0]

    # Construct the bilinear forms
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
    a_ode                   = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        a_tmp = ngs.BilinearForm(model.fes)

        a_tmp += a_lst[i]
        a_tmp += a_ode[i]
        a.append(a_tmp)

    # Construct the linear form
    L : List[BilinearForm]  = []
    L_lst                   = model.construct_linear(V, gfu_lst[0], tmp_dt, 0)
    L_imex_1                = model.construct_imex_explicit(V, gfu_lst[1], tmp_dt, 1)
    L_imex_2                = model.construct_imex_explicit(V, gfu_lst[2], tmp_dt, 2)
    L_imex_3                = model.construct_imex_explicit(V, gfu_lst[3], tmp_dt, 3)
    for i in range(model.num_weak_forms):
        L_tmp = ngs.LinearForm(model.fes)

        L_tmp += L_lst[i]
        L_tmp += 3.0 * L_imex_1[i]
        L_tmp += -3.0 * L_imex_2[i]
        L_tmp += L_imex_3[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'SBDF')

    return a, L


def adaptive_IMEX_pred(model: Model, gfu_0: List[GridFunction], dt: List[Parameter]) \
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Predictor for the adaptive time-stepping IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of previous time steps ordered from most recent to oldest.
        dt: List of timestep sizes ordered from most recent to oldest.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """
    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # Create list of the previous time-step's result
    # NOTE: It is assumed that pressure is always the 2nd in the list
    gfu_prev = [gfu_lst[0][i] for i in range(len(gfu_lst[0]))]
    # Zero out pressure
    gfu_prev[1] = 0.0

    # Operators specific to this time integration scheme.
    w = dt[0] / dt[1]
    E = model.construct_gfu().components[0]
    E_expr = (1.0 + w) * gfu_lst[0][0] - w * gfu_lst[1][0]
    E.Set(E_expr)

    # All the terms use the same dt.
    tmp_dt = dt[0]

    # Construct the bilinear forms
    a: List[BilinearForm]   = []
    a_lst                   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
    a_ode                   = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        a_tmp = BilinearForm(model.fes)

        a_tmp += a_lst[i]
        a_tmp += a_ode[i]

        a.append(a_tmp)

    # Construct the linear forms
    L : List[BilinearForm]  = []
    # TODO: Why doesn't using E work? Numerical error?
    L_lst                   = model.construct_linear(V, gfu_lst[0], tmp_dt, 0)
    L_imex                  = model.construct_imex_explicit(V, gfu_prev, tmp_dt, 0)
    for i in range(model.num_weak_forms):
        L_tmp = LinearForm(model.fes)

        L_tmp += L_lst[i]
        L_tmp += L_imex[i]

        L.append(L_tmp)

    # Add time discretization terms
    _add_dt_terms(a, L, gfu_lst, model, 'crank nicolson')

    return a, L


def RK_222(model: Model, gfu_0: List[GridFunction], dt: List[Parameter], step: int) \
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Two-step second-order Runge Kutta IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.
    
    Args:
        model: The model to solve.
        gfu_0: List of the solutions of intermediate steps in reverse step order (ie: [step n+1 sol, step 2 sol, step 1
            sol, step n sol]).
        dt: List of current time step size.
        step: Which step of the Runge Kutta scheme should be assembled.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """

    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # Define scheme-specific constants. These might eventually become user-specified constants.
    gamma = 0.5 * (2.0 - ngs.sqrt(2.0))
    delta = 1.0 - 1.0 / (2.0 - ngs.sqrt(2.0))

    # Define the variables to hold the linear and bilinear forms
    a: List[BilinearForm] = []
    L: List[BilinearForm] = []

    if step == 1:
        # All the terms use the same dt.
        tmp_dt = dt[1]

        # Construct the bilinear forms
        a_lst = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 1)
        a_ode = model.construct_bilinear_time_ODE(U, V, tmp_dt, 1)
        for i in range(model.num_weak_forms):
            a_tmp = BilinearForm(model.fes)

            a_tmp += a_lst[i]
            a_tmp += a_ode[i]

            a.append(a_tmp)

        # Construct the linear forms
        L_lst   = model.construct_linear(V, None, tmp_dt, 1)
        L_imex  = model.construct_imex_explicit(V, gfu_lst[2], tmp_dt, 2)
        for i in range(model.num_weak_forms):
            L_tmp = LinearForm(model.fes)

            L_tmp += L_lst[i]
            L_tmp += L_imex[i]

            L.append(L_tmp)
    elif step == 2:
        # All the terms use the same dt.
        tmp_dt = dt[0]

        # Construct the bilinear forms
        a_lst   = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
        a_ode   = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
        for i in range(model.num_weak_forms):
            a_tmp = BilinearForm(model.fes)

            a_tmp += a_lst[i]
            a_tmp += gamma * a_ode[i]

            a.append(a_tmp)

        # Construct the linear form
        L_lst_0     = model.construct_linear(V, gfu_lst[0], tmp_dt, 0)
        L_lst_1     = model.construct_linear(V, gfu_lst[1], tmp_dt, 1)
        L_imex_1    = model.construct_imex_explicit(V, gfu_lst[1], tmp_dt, 1)
        L_imex_2    = model.construct_imex_explicit(V, gfu_lst[2], tmp_dt, 2)
        L_ode       = model.construct_bilinear_time_ODE(gfu_lst[1], V, tmp_dt, 1)
        for i in range(model.num_weak_forms):
            L_tmp = LinearForm(model.fes)

            L_tmp += gamma          * L_lst_0[i]
            L_tmp += (1.0 - gamma)  * L_lst_1[i]
            L_tmp += (1.0 - delta)  * L_imex_1[i]
            L_tmp += delta          * L_imex_2[i]
            L_tmp += -(1.0 - gamma) * L_ode[i]

            L.append(L_tmp)
    else:
        raise ValueError('RK 222 only has 2 steps. Step \'{}\' is invalid'.format(step))

    _add_dt_terms(a, L, gfu_lst, model, 'RK 222')

    return a, L


def RK_232(model: Model, gfu_0: List[GridFunction], dt: List[Parameter], step: int)\
        -> Tuple[List[BilinearForm], List[LinearForm]]:
    """
    Three-step second-order Runge Kutta IMEX time integration scheme.

    This function constructs the final bilinear and linear forms for the time integration scheme by adding the necessary
    time-dependent terms to the model's stationary terms. The returned bilinear and linear forms have NOT been
    assembled.

    Args:
        model: The model to solve.
        gfu_0: List of the solutions of intermediate steps in reverse step order (ie: [step n+1 sol, step 2 sol, step 1
            sol, step n sol]).
        dt: List of current time step size.
        step: Which step of the Runge Kutta scheme should be assembled.

    Returns:
        Tuple[BilinearForm, LinearForm]:
            - a: A list of the final bilinear forms (as a BilinearForm but not assembled).
            - L: A list of the final linear forms (as a LinearForm but not assembled).
    """

    U, V = model.get_trial_and_test_functions()

    gfu_lst = _split_gfu(gfu_0)

    # Define scheme-specific constants. These might eventually become user-specified constants.
    gamma = 0.5 * (2.0 - ngs.sqrt(2.0))
    delta = -2.0 * ngs.sqrt(2.0) / 3.0

    # Define the variables to hold the linear and bilinear forms
    a: List[BilinearForm] = []
    L: List[BilinearForm] = []

    if step == 1:
        # All the terms use the same dt.
        tmp_dt = dt[2]

        # Construct the bilinear form
        a_lst = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 2)
        a_ode = model.construct_bilinear_time_ODE(U, V, tmp_dt, 2)
        for i in range(model.num_weak_forms):
            a_tmp = BilinearForm(model.fes)

            a_tmp += a_lst[i]
            a_tmp += a_ode[i]

            a.append(a_tmp)

        # Construct the linear form
        L_lst   = model.construct_linear(V, None, tmp_dt, 2)
        L_imex  = model.construct_imex_explicit(V, gfu_lst[3], tmp_dt, 3)
        for i in range(model.num_weak_forms):
            L_tmp = LinearForm(model.fes)

            L_tmp += L_lst[i]
            L_tmp += L_imex[i]

            L.append(L_tmp)
    elif step == 2:
        # All the terms use the same dt.
        tmp_dt = dt[1]

        # Construct the bilinear form
        a_lst = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 1)
        a_ode = model.construct_bilinear_time_ODE(U, V, tmp_dt, 1)
        for i in range(model.num_weak_forms):
            a_tmp = BilinearForm(model.fes)

            a_tmp += a_lst[i]
            a_tmp += gamma * a_ode[i]

            a.append(a_tmp)

        # Construct the linear form
        L_lst_1     = model.construct_linear(V, None, tmp_dt, 1)
        L_lst_2     = model.construct_linear(V, gfu_lst[2], tmp_dt, 2)
        L_imex_2    = model.construct_imex_explicit(V, gfu_lst[2], tmp_dt, 2)
        L_imex_3    = model.construct_imex_explicit(V, gfu_lst[3], tmp_dt, 3)
        L_ode       = model.construct_bilinear_time_ODE(gfu_lst[2], V, tmp_dt, 2)
        for i in range(model.num_weak_forms):
            L_tmp = LinearForm(model.fes)

            L_tmp += gamma          * L_lst_1[i]
            L_tmp += (1.0 - gamma)  * L_lst_2[i]
            L_tmp += (1.0 - delta)  * L_imex_2[i]
            L_tmp += delta          * L_imex_3[i]
            L_tmp += -(1.0 - gamma) * L_ode[i]

            L.append(L_tmp)
    elif step == 3:
        # All the terms use the same dt.
        tmp_dt = dt[0]

        # Construct the bilinear form
        a_lst = model.construct_bilinear_time_coefficient(U, V, tmp_dt, 0)
        a_ode = model.construct_bilinear_time_ODE(U, V, tmp_dt, 0)
        for i in range(model.num_weak_forms):
            a_tmp = BilinearForm(model.fes)

            a_tmp += a_lst[i]
            a_tmp += gamma * a_ode[i]

            a.append(a_tmp)

        # Construct the linear form
        L_lst_0     = model.construct_linear(V, gfu_lst[0], tmp_dt, 0)
        L_lst_2     = model.construct_linear(V, gfu_lst[2], tmp_dt, 2)
        L_imex_1    = model.construct_imex_explicit(V, gfu_lst[1], tmp_dt, 1)
        L_imex_2    = model.construct_imex_explicit(V, gfu_lst[2], tmp_dt, 2)
        L_ode       = model.construct_bilinear_time_ODE(gfu_lst[2], V, tmp_dt, 2)

        for i in range(model.num_weak_forms):
            L_tmp = LinearForm(model.fes)

            L_tmp += gamma          * L_lst_0[i]
            L_tmp += (1.0 - gamma)  * L_lst_2[i]
            L_tmp += delta          * L_imex_1[i]
            L_tmp += (1.0 - delta)  * L_imex_2[i]
            L_tmp += -(1.0 - gamma) * L_ode[i]

            L.append(L_tmp)
    else:
        raise ValueError('RK 232 only has 3 steps. Step \'{}\' is invalid'.format(step))

    _add_dt_terms(a, L, gfu_lst, model, 'RK 232')

    return a, L


def _add_dt_terms(a: List[BilinearForm], L: List[BilinearForm], gfu_lst: List[List[GridFunction]],
                  model: Model, time_discretization_scheme: str) -> None:
    """
    Function to handle the adding of the time discretization terms to the linear and bilinear forms.

    Args:
        a: List of all of the bilinear forms for the model
        L: List of all of the linear forms for the model
        gfu_lst: List of the solutions of previous time steps in reverse chronological order.
        model:
        time_discretization_scheme: The name of the time integration discretization being used.
    """
    # Construct the time discretization terms
    a_dt, L_dt = model.time_derivative_terms(gfu_lst, time_discretization_scheme)
    # When adding the time discretization term, multiply by the phase field if using the diffuse interface method.
    if model.DIM:
        for i in range(len(a_dt)):
            a_dt[i] *= model.DIM_solver.phi_gfu
            L_dt[i] *= model.DIM_solver.phi_gfu

    for i in range(model.num_weak_forms):
        a[i] += a_dt[i] * dx
        L[i] += L_dt[i] * dx


def _split_gfu(gfu: List[GridFunction]) -> List[List[GridFunction]]:
    """
    Function to separate out the various components of each gridfunction solution to a previous timestep.

    The first entry in gfu_lst is None so that its indexing matches the indexing for dt and the step argument in
    the functions for constructing the weak form (ie: gfu_lst[1], dt[1] and step=1 all refer to values at time n).
    The final result should be gfu_lst = [None, [component 0, component 1...] at t^n, [component 0, component 1...] 
    at t^n-1, ...].

    Args:
        gfu: List of gridfunctions to split up

    Returns:
        [[component 0, component 1...] at t^n, [component 0, component 1...] at t^n-1, ...]
    """

    # NOTE: First entry is supposed to be None, see function docstring
    gfu_lst: List[List[GridFunction]] = [None]

    for i in range(len(gfu)):
        if len(gfu[i].components) > 0:
            gfu_lst.append([gfu[i].components[j] for j in range(len(gfu[i].components))])
        else:
            gfu_lst.append([gfu[i]])

    return gfu_lst
