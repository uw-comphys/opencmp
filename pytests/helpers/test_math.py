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

from pytest import fixture
from opencmp.helpers.math import tanh, sig, H_t, H_s
from numpy import isclose
from ngsolve import Parameter, CoefficientFunction, Mesh
from typing import Tuple
from opencmp.config_functions.expanded_config_parser import ConfigParser
from opencmp.helpers.io import load_mesh


@fixture
def simple_mesh() -> Mesh:
    """
    Function to return a small simple mesh file.

    Returns:
        The mesh.
    """

    # Create config parser
    c = ConfigParser('pytests/helpers/mesh/config_blank')
    c['MESH'] = {'filename': 'pytests/mesh_files/unit_square_coarse.vol'}

    # Load the mesh
    mesh = load_mesh(c)

    return mesh


def _val_from_coeff(c: CoefficientFunction, mesh: Mesh) -> Tuple[float]:
    """
    Helper function to evaluate a coefficient function and return values as a tuple.

    | If coefficient function is a scalar, a tuple of length 1 is returned. Otherwise a tuple containing each vector
    | element, in order, is returned.
    |
    | NOTE: It is assumed that the coefficient function does not vary in space.

    Args:
         c: The coefficient function to evaluate.

    Returns:
        Tuple containing the values of the coefficient function.
    """
    val = c(mesh(0, 0))

    # Convert
    if type(val) is float:
        val = tuple([val])

    return val


class TestTanh:
    """
    Test the implementation of the hyperbolic tangent implementation.
    """
    def test_0(self, simple_mesh: Mesh):
        num_val   = tanh(0)
        param_res = tanh(Parameter(0))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf(self, simple_mesh: Mesh):
        num_val   = tanh(1e30)
        param_res = tanh(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,  1)
        assert isclose(param_val, 1)

    def test_neg_inf(self, simple_mesh: Mesh):
        num_val   = tanh(-1e30)
        param_res = tanh(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   -1)
        assert isclose(param_val, -1)


class TestSig:
    """
    Tests the implementation of the sigmoid function.
    """
    def test_0(self, simple_mesh: Mesh):
        num_val   = sig(0)
        param_res = sig(Parameter(0))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0.5)
        assert isclose(param_val, 0.5)

    def test_neg_inf_num(self, simple_mesh: Mesh):
        num_val   = sig(-1e30)
        param_res = sig(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf_num(self, simple_mesh: Mesh):
        num_val   = sig(1e30)
        param_res = sig(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   1)
        assert isclose(param_val, 1)


class TestHT:
    """
    Test the tanh-based approximated Heaviside function.
    """
    def test_neg_inf_num(self, simple_mesh: Mesh):
        num_val   = H_t(-1e30)
        param_res = H_t(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf_num(self, simple_mesh: Mesh):
        num_val  = H_t(1e30)
        param_res = H_t(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   1)
        assert isclose(param_val, 1)

    def test_width(self, simple_mesh: Mesh):
        num_val_0   = H_t(0)
        param_res_0 = H_t(Parameter(0))
        param_val_0 = _val_from_coeff(param_res_0, simple_mesh)[0]

        num_val_1   = H_t(0.1)
        param_res_1 = H_t(Parameter(0.1))
        param_val_1 = _val_from_coeff(param_res_1, simple_mesh)[0]

        assert isclose(num_val_0,   0, atol=0.0001)
        assert isclose(param_val_0, 0, atol=0.0001)
        assert isclose(num_val_1,   1, atol=0.0001)
        assert isclose(param_val_1, 1, atol=0.0001)

    def test_0(self, simple_mesh: Mesh):
        num_val   = H_t(0.05)
        param_res = H_t(Parameter(0.05))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0.5)
        assert isclose(param_val, 0.5)

    def test_shift(self, simple_mesh: Mesh):
        t = Parameter(0)

        result_0 = H_t(t - 1)
        result_1 = H_t(t + 1)

        result_0 = _val_from_coeff(result_0, simple_mesh)
        result_1 = _val_from_coeff(result_1, simple_mesh)

        assert isclose(result_0, 0)
        assert isclose(result_1, 1)

    def test_time_varying(self, simple_mesh: Mesh):
        t = Parameter(-1)

        h = H_t(t)

        result_0 = _val_from_coeff(h, simple_mesh)

        t.Set(t.Get() + 1.05)
        result_05 = _val_from_coeff(h, simple_mesh)

        t.Set(t.Get() + 0.95)
        result_1 = _val_from_coeff(h, simple_mesh)

        assert isclose(result_0,  0)
        assert isclose(result_05, 0.5)
        assert isclose(result_1,  1)


class TestHS:
    """
    Test the sigmoid-based approximated Heaviside function.
    """
    def test_neg_inf_num(self, simple_mesh: Mesh):
        num_val   = H_s(-1e30)
        param_res = H_s(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf_num(self, simple_mesh: Mesh):
        num_val  = H_s(1e30)
        param_res = H_s(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   1)
        assert isclose(param_val, 1)

    def test_width(self, simple_mesh: Mesh):
        num_val_0   = H_s(0)
        param_res_0 = H_s(Parameter(0))
        param_val_0 = _val_from_coeff(param_res_0, simple_mesh)[0]

        num_val_1   = H_s(0.1)
        param_res_1 = H_s(Parameter(0.1))
        param_val_1 = _val_from_coeff(param_res_1, simple_mesh)[0]

        assert isclose(num_val_0,   0, atol=0.0001)
        assert isclose(param_val_0, 0, atol=0.0001)
        assert isclose(num_val_1,   1, atol=0.0001)
        assert isclose(param_val_1, 1, atol=0.0001)

    def test_0(self, simple_mesh: Mesh):
        num_val   = H_s(0.05)
        param_res = H_s(Parameter(0.05))
        param_val = _val_from_coeff(param_res, simple_mesh)[0]

        assert isclose(num_val,   0.5)
        assert isclose(param_val, 0.5)

    def test_shift(self, simple_mesh: Mesh):
        t = Parameter(0)

        result_0 = H_s(t - 1)
        result_1 = H_s(t + 1)

        result_0 = _val_from_coeff(result_0, simple_mesh)
        result_1 = _val_from_coeff(result_1, simple_mesh)

        assert isclose(result_0, 0)
        assert isclose(result_1, 1)

    def test_time_varying(self, simple_mesh: Mesh):
        t = Parameter(-1)

        h = H_s(t)

        result_0 = _val_from_coeff(h, simple_mesh)

        t.Set(t.Get() + 1.05)
        result_05 = _val_from_coeff(h, simple_mesh)

        t.Set(t.Get() + 0.95)
        result_1 = _val_from_coeff(h, simple_mesh)

        assert isclose(result_0,  0)
        assert isclose(result_05, 0.5)
        assert isclose(result_1,  1)
