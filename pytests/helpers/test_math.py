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
from opencmp.helpers.math import tanh, sig, H_t, H_s, Max, Min
from numpy import isclose
from ngsolve import Parameter, CoefficientFunction, Mesh, x
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


def _val_from_coeff(c: CoefficientFunction, mesh: Mesh, x, y) -> Tuple[float]:
    """
    Helper function to evaluate a coefficient function and return values as a tuple.

    | If coefficient function is a scalar, a tuple of length 1 is returned. Otherwise a tuple containing each vector
    | element, in order, is returned.
    |
    | NOTE: It is assumed that the coefficient function does not vary in space.

    Args:
         c:     The coefficient function to evaluate.
         mesh:  The mesh to evaluate on
         x:     The x-coordinate to evaluate at
         y:     The y-coordinate to evaluate at

    Returns:
        Tuple containing the values of the coefficient function.
        :param x:
        :param y:
    """
    val = c(mesh(x, y))

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
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf(self, simple_mesh: Mesh):
        num_val   = tanh(1e30)
        param_res = tanh(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,  1)
        assert isclose(param_val, 1)

    def test_neg_inf(self, simple_mesh: Mesh):
        num_val   = tanh(-1e30)
        param_res = tanh(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   -1)
        assert isclose(param_val, -1)


class TestSig:
    """
    Tests the implementation of the sigmoid function.
    """
    def test_0(self, simple_mesh: Mesh):
        num_val   = sig(0)
        param_res = sig(Parameter(0))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0.5)
        assert isclose(param_val, 0.5)

    def test_neg_inf_num(self, simple_mesh: Mesh):
        num_val   = sig(-1e30)
        param_res = sig(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf_num(self, simple_mesh: Mesh):
        num_val   = sig(1e30)
        param_res = sig(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   1)
        assert isclose(param_val, 1)


class TestHT:
    """
    Test the tanh-based approximated Heaviside function.
    """
    def test_neg_inf_num(self, simple_mesh: Mesh):
        num_val   = H_t(-1e30)
        param_res = H_t(Parameter(-1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf_num(self, simple_mesh: Mesh):
        num_val  = H_t(1e30)
        param_res = H_t(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   1)
        assert isclose(param_val, 1)

    def test_width(self, simple_mesh: Mesh):
        num_val_0   = H_t(0)
        param_res_0 = H_t(Parameter(0))
        param_val_0 = _val_from_coeff(param_res_0, simple_mesh, 0, 0)[0]

        num_val_1   = H_t(0.1)
        param_res_1 = H_t(Parameter(0.1))
        param_val_1 = _val_from_coeff(param_res_1, simple_mesh, 0, 0)[0]

        assert isclose(num_val_0,   0, atol=0.0001)
        assert isclose(param_val_0, 0, atol=0.0001)
        assert isclose(num_val_1,   1, atol=0.0001)
        assert isclose(param_val_1, 1, atol=0.0001)

    def test_0(self, simple_mesh: Mesh):
        num_val   = H_t(0.05)
        param_res = H_t(Parameter(0.05))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0.5)
        assert isclose(param_val, 0.5)

    def test_shift(self, simple_mesh: Mesh):
        t = Parameter(0)

        result_0 = H_t(t - 1)
        result_1 = H_t(t + 1)

        result_0 = _val_from_coeff(result_0, simple_mesh, 0, 0)
        result_1 = _val_from_coeff(result_1, simple_mesh, 0, 0)

        assert isclose(result_0, 0)
        assert isclose(result_1, 1)

    def test_time_varying(self, simple_mesh: Mesh):
        t = Parameter(-1)

        h = H_t(t)

        result_0 = _val_from_coeff(h, simple_mesh, 0, 0)

        t.Set(t.Get() + 1.05)
        result_05 = _val_from_coeff(h, simple_mesh, 0, 0)

        t.Set(t.Get() + 0.95)
        result_1 = _val_from_coeff(h, simple_mesh, 0, 0)

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
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0)
        assert isclose(param_val, 0)

    def test_pos_inf_num(self, simple_mesh: Mesh):
        num_val  = H_s(1e30)
        param_res = H_s(Parameter(1e30))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   1)
        assert isclose(param_val, 1)

    def test_width(self, simple_mesh: Mesh):
        num_val_0   = H_s(0)
        param_res_0 = H_s(Parameter(0))
        param_val_0 = _val_from_coeff(param_res_0, simple_mesh, 0, 0)[0]

        num_val_1   = H_s(0.1)
        param_res_1 = H_s(Parameter(0.1))
        param_val_1 = _val_from_coeff(param_res_1, simple_mesh, 0, 0)[0]

        assert isclose(num_val_0,   0, atol=0.0001)
        assert isclose(param_val_0, 0, atol=0.0001)
        assert isclose(num_val_1,   1, atol=0.0001)
        assert isclose(param_val_1, 1, atol=0.0001)

    def test_0(self, simple_mesh: Mesh):
        num_val   = H_s(0.05)
        param_res = H_s(Parameter(0.05))
        param_val = _val_from_coeff(param_res, simple_mesh, 0, 0)[0]

        assert isclose(num_val,   0.5)
        assert isclose(param_val, 0.5)

    def test_shift(self, simple_mesh: Mesh):
        t = Parameter(0)

        result_0 = H_s(t - 1)
        result_1 = H_s(t + 1)

        result_0 = _val_from_coeff(result_0, simple_mesh, 0, 0)
        result_1 = _val_from_coeff(result_1, simple_mesh, 0, 0)

        assert isclose(result_0, 0)
        assert isclose(result_1, 1)

    def test_time_varying(self, simple_mesh: Mesh):
        t = Parameter(-1)

        h = H_s(t)

        result_0 = _val_from_coeff(h, simple_mesh, 0, 0)

        t.Set(t.Get() + 1.05)
        result_05 = _val_from_coeff(h, simple_mesh, 0, 0)

        t.Set(t.Get() + 0.95)
        result_1 = _val_from_coeff(h, simple_mesh, 0, 0)

        assert isclose(result_0,  0)
        assert isclose(result_05, 0.5)
        assert isclose(result_1,  1)


class TestMax:
    """
    Test that the Max function is working properly
    """

    def test_both_pos(self, simple_mesh: Mesh):
        a = 10.0
        b = CoefficientFunction(9.5 + x)

        max_a_first = Max(a, b)
        max_b_first = Max(b, a)

        max_left_a  = _val_from_coeff(max_a_first, simple_mesh, 0,   0)[0]
        max_left_b  = _val_from_coeff(max_b_first, simple_mesh, 0,   0)[0]
        max_mid_a   = _val_from_coeff(max_a_first, simple_mesh, 0.5, 0)[0]
        max_mid_b   = _val_from_coeff(max_b_first, simple_mesh, 0.5, 0)[0]
        max_right_a = _val_from_coeff(max_a_first, simple_mesh, 1,   0)[0]
        max_right_b = _val_from_coeff(max_b_first, simple_mesh, 1,   0)[0]

        assert isclose(10, max_left_a)
        assert isclose(max_left_a, max_left_b)

        assert isclose(10, max_mid_a)
        assert isclose(max_mid_a, max_mid_b)

        assert isclose(10.5, max_right_a)
        assert isclose(max_right_a, max_right_b)

    def test_one_pos_one_neg(self, simple_mesh: Mesh):
        a = 10.0
        b = CoefficientFunction(-9.5 + x)

        max_a_first = Max(a, b)
        max_b_first = Max(b, a)

        max_left_a = _val_from_coeff(max_a_first, simple_mesh, 0, 0)[0]
        max_left_b = _val_from_coeff(max_b_first, simple_mesh, 0, 0)[0]
        max_mid_a = _val_from_coeff(max_a_first, simple_mesh, 0.5, 0)[0]
        max_mid_b = _val_from_coeff(max_b_first, simple_mesh, 0.5, 0)[0]
        max_right_a = _val_from_coeff(max_a_first, simple_mesh, 1, 0)[0]
        max_right_b = _val_from_coeff(max_b_first, simple_mesh, 1, 0)[0]

        assert isclose(10, max_left_a)
        assert isclose(max_left_a, max_left_b)

        assert isclose(10, max_mid_a)
        assert isclose(max_mid_a, max_mid_b)

        assert isclose(10, max_right_a)
        assert isclose(max_right_a, max_right_b)

    def test_one_zero(self, simple_mesh: Mesh):
        a = 0.0
        b = CoefficientFunction(x - 0.5)

        max_a_first = Max(a, b)
        max_b_first = Max(b, a)

        max_left_a  = _val_from_coeff(max_a_first, simple_mesh, 0,   0)[0]
        max_left_b  = _val_from_coeff(max_b_first, simple_mesh, 0,   0)[0]
        max_mid_a   = _val_from_coeff(max_a_first, simple_mesh, 0.5, 0)[0]
        max_mid_b   = _val_from_coeff(max_b_first, simple_mesh, 0.5, 0)[0]
        max_right_a = _val_from_coeff(max_a_first, simple_mesh, 1,   0)[0]
        max_right_b = _val_from_coeff(max_b_first, simple_mesh, 1,   0)[0]

        assert isclose(0, max_left_a)
        assert isclose(max_left_a, max_left_b)

        assert isclose(0, max_mid_a)
        assert isclose(max_mid_a, max_mid_b)

        assert isclose(0.5, max_right_a)
        assert isclose(max_right_a, max_right_b)

    def test_both_neg(self, simple_mesh: Mesh):
        a = -10.0
        b = CoefficientFunction(-9.5 - x)

        max_a_first = Max(a, b)
        max_b_first = Max(b, a)

        max_left_a  = _val_from_coeff(max_a_first, simple_mesh, 0,   0)[0]
        max_left_b  = _val_from_coeff(max_b_first, simple_mesh, 0,   0)[0]
        max_mid_a   = _val_from_coeff(max_a_first, simple_mesh, 0.5, 0)[0]
        max_mid_b   = _val_from_coeff(max_b_first, simple_mesh, 0.5, 0)[0]
        max_right_a = _val_from_coeff(max_a_first, simple_mesh, 1,   0)[0]
        max_right_b = _val_from_coeff(max_b_first, simple_mesh, 1,   0)[0]

        assert isclose(-9.5, max_left_a)
        assert isclose(max_left_a, max_left_b)

        assert isclose(-10, max_mid_a)
        assert isclose(max_mid_a, max_mid_b)

        assert isclose(-10, max_right_a)
        assert isclose(max_right_a, max_right_b)


class TestMin:
    """
    Test that the Min function is working properly
    """

    def test_both_pos(self, simple_mesh: Mesh):
        a = 10.0
        b = CoefficientFunction(9.5 + x)

        min_a_first = Min(a, b)
        min_b_first = Min(b, a)

        min_left_a  = _val_from_coeff(min_a_first, simple_mesh, 0, 0)[0]
        min_left_b  = _val_from_coeff(min_b_first, simple_mesh, 0, 0)[0]
        min_mid_a   = _val_from_coeff(min_a_first, simple_mesh, 0.5, 0)[0]
        min_mid_b   = _val_from_coeff(min_b_first, simple_mesh, 0.5, 0)[0]
        min_right_a = _val_from_coeff(min_a_first, simple_mesh, 1, 0)[0]
        min_right_b = _val_from_coeff(min_b_first, simple_mesh, 1, 0)[0]

        assert isclose(9.5, min_left_a)
        assert isclose(min_left_a, min_left_b)

        assert isclose(10, min_mid_a)
        assert isclose(min_mid_a, min_mid_b)

        assert isclose(10, min_right_a)
        assert isclose(min_right_a, min_right_b)

    def test_one_neg(self, simple_mesh: Mesh):
        a = 10.0
        b = CoefficientFunction(-9.5 - x)

        min_a_first = Min(a, b)
        min_b_first = Min(b, a)

        min_left_a = _val_from_coeff(min_a_first, simple_mesh, 0, 0)[0]
        min_left_b = _val_from_coeff(min_b_first, simple_mesh, 0, 0)[0]
        min_mid_a = _val_from_coeff(min_a_first, simple_mesh, 0.5, 0)[0]
        min_mid_b = _val_from_coeff(min_b_first, simple_mesh, 0.5, 0)[0]
        min_right_a = _val_from_coeff(min_a_first, simple_mesh, 1, 0)[0]
        min_right_b = _val_from_coeff(min_b_first, simple_mesh, 1, 0)[0]

        assert isclose(-9.5, min_left_a)
        assert isclose(min_left_a, min_left_b)

        assert isclose(-10, min_mid_a)
        assert isclose(min_mid_a, min_mid_b)

        assert isclose(-10.5, min_right_a)
        assert isclose(min_right_a, min_right_b)

    def test_one_zero(self, simple_mesh: Mesh):
        a = 0.0
        b = CoefficientFunction(x - 0.5)

        min_a_first = Min(a, b)
        min_b_first = Min(b, a)

        min_left_a = _val_from_coeff(min_a_first, simple_mesh, 0, 0)[0]
        min_left_b = _val_from_coeff(min_b_first, simple_mesh, 0, 0)[0]
        min_mid_a = _val_from_coeff(min_a_first, simple_mesh, 0.5, 0)[0]
        min_mid_b = _val_from_coeff(min_b_first, simple_mesh, 0.5, 0)[0]
        min_right_a = _val_from_coeff(min_a_first, simple_mesh, 1, 0)[0]
        min_right_b = _val_from_coeff(min_b_first, simple_mesh, 1, 0)[0]

        assert isclose(-0.5, min_left_a)
        assert isclose(min_left_a, min_left_b)

        assert isclose(0, min_mid_a)
        assert isclose(min_mid_a, min_mid_b)

        assert isclose(0, min_right_a)
        assert isclose(min_right_a, min_right_b)

    def test_both_neg(self, simple_mesh: Mesh):
        a = -10.0
        b = CoefficientFunction(-9.5 - x)

        min_a_first = Min(a, b)
        min_b_first = Min(b, a)

        min_left_a = _val_from_coeff(min_a_first, simple_mesh, 0, 0)[0]
        min_left_b = _val_from_coeff(min_b_first, simple_mesh, 0, 0)[0]
        min_mid_a = _val_from_coeff(min_a_first, simple_mesh, 0.5, 0)[0]
        min_mid_b = _val_from_coeff(min_b_first, simple_mesh, 0.5, 0)[0]
        min_right_a = _val_from_coeff(min_a_first, simple_mesh, 1, 0)[0]
        min_right_b = _val_from_coeff(min_b_first, simple_mesh, 1, 0)[0]

        assert isclose(-10, min_left_a)
        assert isclose(min_left_a, min_left_b)

        assert isclose(-10, min_mid_a)
        assert isclose(min_mid_a, min_mid_b)

        assert isclose(-10.5, min_right_a)
        assert isclose(min_right_a, min_right_b)
