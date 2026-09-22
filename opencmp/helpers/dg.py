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


def weighted_grad_avg(q: CoefficientFunction, c: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of the gradient of a field weighted by a (possibly discontinuous) coefficient.

    Args:
        q: The field.
        c: The coefficient weighting the gradient on each side of the facet.

    Returns:
        The average of c * Grad(q) at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        return 0.5 * (c * Grad(q) + c.Other() * Grad(q.Other()))
    else:
        return 0.5 * (c * Grad(q) + c.Other() * Grad(q).Other())


def weighted_trans_grad_avg(q: CoefficientFunction, c: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of the transposed gradient of a field weighted by a (possibly discontinuous) coefficient.

    Note this is NOT weighted_grad_avg(q, c).trans in general: this weights on the right,
    (Grad(q) * c)^T, and the two only coincide for scalar c.

    Args:
        q: The field.
        c: The coefficient weighting the gradient on each side of the facet.

    Returns:
        The average of (Grad(q) * c)^T at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        return 0.5 * ((Grad(q) * c).trans + (Grad(q.Other()) * c.Other()).trans)
    else:
        return 0.5 * ((Grad(q) * c).trans + (Grad(q).Other() * c.Other()).trans)


def weighted_div_avg(q: CoefficientFunction, c: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of the divergence of a field weighted by a (possibly discontinuous) coefficient.

    Args:
        q: The field.
        c: The coefficient weighting the divergence on each side of the facet.

    Returns:
        The average of c * div(q) at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        div_q = sum(Grad(q)[i, i] for i in range(q.dim))
        div_q_other = sum(Grad(q.Other())[i, i] for i in range(q.dim))
    else:
        div_q = sum(Grad(q)[i, i] for i in range(q.dim))
        div_q_other = div_q.Other()

    return 0.5 * (div_q * c + div_q_other * c.Other())
