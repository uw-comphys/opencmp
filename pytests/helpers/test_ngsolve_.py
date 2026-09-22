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
from netgen.csg import unit_cube
from numpy import allclose
from pytest import fixture

from opencmp.helpers.ngsolve_ import curl_3d


@fixture
def cube_mesh() -> ngs.Mesh:
    """A coarse unit-cube mesh."""

    return ngs.Mesh(unit_cube.GenerateMesh(maxh=0.4))


def test_curl_3d(cube_mesh) -> None:
    """
    curl_3d against a field whose curl is known analytically.

    F = (x*y + 2*z^2, 3*x^2 - y*z, x*z + 5*y^2)  ->  curl(F) = (11*y, 3*z, 5*x).

    Every component of the expected curl carries a different coefficient AND a
    different variable, so a permuted or sign-flipped gradient index cannot pass
    by coincidence. F is quadratic, so an order-2 space represents it exactly and
    the comparison holds to machine precision.
    """

    fes = ngs.H1(cube_mesh, order=2, dim=3)
    gfu = ngs.GridFunction(fes)
    gfu.Set(ngs.CoefficientFunction((ngs.x * ngs.y + 2 * ngs.z ** 2,
                                     3 * ngs.x ** 2 - ngs.y * ngs.z,
                                     ngs.x * ngs.z + 5 * ngs.y ** 2)))

    expected = ngs.CoefficientFunction((11 * ngs.y, 3 * ngs.z, 5 * ngs.x))

    for point in [(0.3, 0.4, 0.6), (0.15, 0.85, 0.25), (0.5, 0.5, 0.5)]:
        mip = cube_mesh(*point)
        assert allclose(curl_3d(gfu)(mip), expected(mip), atol=1e-10)


def test_curl_3d_of_a_gradient_is_zero(cube_mesh) -> None:
    """curl(grad(phi)) == 0 for any smooth phi -- an identity, not a hand-computed value."""

    phi = ngs.x ** 2 * ngs.y + ngs.y * ngs.z ** 2 + 3 * ngs.x * ngs.z

    fes = ngs.H1(cube_mesh, order=2, dim=3)
    gfu = ngs.GridFunction(fes)
    # grad(phi) written out componentwise so the test does not depend on the same
    # gradient-flattening convention curl_3d is being tested for.
    gfu.Set(ngs.CoefficientFunction((2 * ngs.x * ngs.y + 3 * ngs.z,
                                     ngs.x ** 2 + ngs.z ** 2,
                                     2 * ngs.y * ngs.z + 3 * ngs.x)))

    for point in [(0.3, 0.4, 0.6), (0.7, 0.2, 0.1)]:
        mip = cube_mesh(*point)
        assert allclose(curl_3d(gfu)(mip), (0.0, 0.0, 0.0), atol=1e-10)
