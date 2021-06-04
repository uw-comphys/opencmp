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

from typing import Union
from ngsolve import CoefficientFunction, Parameter, exp, IfPos, cos
from math import pi


def tanh(x: Union[int, float, Parameter, CoefficientFunction]) -> Union[float, CoefficientFunction]:
    """
    Function that implements the hyperbolic tangent in a way that's compatible with NGSolve Parameter and Coefficient
    objects.

    Args:
        x: The value to evaluate tanh at.

    Return:
        tanh(x)
    """

    # Deal with some math overflow problems
    if type(x) is Parameter or type(x) is CoefficientFunction:
        # Added to ensure that the tanh calculation does not return nan
        x_adj = IfPos(x - 350, 350, x)
    else:  # float or int
        if x > 350:
            x_adj = 350
        else:
            x_adj = x
    return (exp(2 * x_adj) - 1.0) / (exp(2 * x_adj) + 1.0)


def sig(x: Union[int, float, Parameter, CoefficientFunction]) -> Union[float, CoefficientFunction]:
    """
    Function that implements the sigmoid function in a way that's compatible with NGSolve Parameter and Coefficient
    objects.

    Args:
        x: The value to evaluate the sigmoid at

    Return:
        sig(x)
    """

    return 1 / (1 + exp(-x))


def H_t(t: Union[int, float, Parameter, CoefficientFunction], w: float = 0.1) -> Union[float, CoefficientFunction]:
    """
    Function to approximate the Heaviside function (step function) using a hyperbolic tangent (tanh).

    Args:
        t: Parameter reprenting time (or another value)
        w: Length of time over which to go from 0.0001 to 0.9999

    Return:
        The evaluated value of the approximated Heaviside function
    """

    # Transition width must be positive
    assert w > 0

    # Term to shift the tanh left-right so that we get the value we desire
    # NOTE: The 5 shifts tanh so that at t == 0 its value is -0.9999 (four 9s truncated)
    shift = 5.0

    # Value to scale time domain by so that the jump occurs of a period of w
    # Jump is from (1 - 0.9999) to 0.9999 (after scaling)
    # Get my solving: tanh(shift) = tanh(scaling_term * w - shift)
    scaling_term = (2 * shift) / w

    return 1 / 2 * tanh(scaling_term * t - shift) + 0.5


def H_s(t: Union[int, float, Parameter, CoefficientFunction], w: float = 0.1) -> Union[float, CoefficientFunction]:
    """
    Function to approximate the Heaviside function (step function) using a sigmoid.

    Args:
        t: Parameter representing time (or another value)
        w: Length of time over which to go from 0.0001 to 0.9999

    Return:
        The evaluated value of the approximated Heaviside function
    """

    # Transition width must be positive
    assert w > 0

    # Term to shift the sig left-right so that we get the value we desire
    # NOTE: The 5 shifts sig so that at t == 0 its value is (1 - 0.9999) (four 9s truncated)
    shift = 9.21025

    # Value to scale time domain by so that the jump occurs of a period of w
    # Jump is from (1 - 0.9999) to 0.9999.
    # Get my solving: sig(shift) = sig(scaling_term * w - shift)
    scaling_term = (2 * shift) / w

    return sig(scaling_term * t - shift)


def ramp_cos(t: Union[int, float, Parameter], p1: float = 0.0, p2: float = 1.0,
             tr: float = 1.0) -> float:
    """
    Function to ramp from one specified value to another in a specified amount of time using the cosine function.

    Args:
        t: Parameter representing time (or another value).
        p1: The value to start the ramp at.
        p2: The value to end the ramp at.
        tr: How quickly to ramp.

    Return:
        The evaluated value of the ramp function.
    """

    return IfPos(tr - t, 0.5 * (p1 - p2) * cos(t * pi / tr) + 0.5 * (p1 + p2), p2)
