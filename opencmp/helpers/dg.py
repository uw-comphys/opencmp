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

from ngsolve import CoefficientFunction, Grad
from ngsolve.comp import ProxyFunction


def jump(q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the jump of a field.

    Args:
        q: The field.

    Returns:
        The jump of q at every facet of the mesh.
    """

    return q - q.Other()


def grad_jump(q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the jump of the gradient of a field.

    Args:
        q: The field.

    Returns:
        The jump of the gradient of q at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        return Grad(q) - Grad(q.Other())
    else:
        return Grad(q) - Grad(q).Other()


def avg(q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of a scalar field.

    Args:
        q: The scalar field.

    Returns:
        The average of q at every facet of the mesh.
    """

    return 0.5 * (q + q.Other())


def grad_avg(q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of the gradient of a field.

    Args:
        q: The field.

    Returns:
        The average of the gradient of q at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        return 0.5 * (Grad(q) + Grad(q.Other()))
    else:
        return 0.5 * (Grad(q) + Grad(q).Other())
