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

import pytest
from opencmp.config_functions.load_config import parse_str, convert_str_to_dict, load_coefficientfunction_into_gridfunction
import pyparsing
import ngsolve as ngs
from ngsolve import Mesh, FESpace
from netgen.geom2d import unit_square
from netgen.csg import CSGeometry, OrthoBrick, Pnt
import math
import netgen
from typing import Tuple, Dict, List, Union


@pytest.fixture()
def presets_convert_str_to_dict() -> Tuple[
    str, Dict[str, Union[List[int], List[str]]], Dict[str, str], List[Dict[str, int]], List[str]]:
    """
    Function to construct the presets for testing convert_str_to_dict.

    Returns:
        Tuple[str, Dict, Dict, Dict, List[str]]:
            - config_filename: Path to the example config file.
            - correct_dict: Dictionary containing the parameters expected from the example config file.
            - correct_re_parse_dict: Dictionary containing the parameters from the example config file that need to be
                re-parsed.
            - new_variables: Dictionary containing the model variables for the example config file.
            - filetypes: List of filetypes to parse as file paths from the example config file.
    """

    config_filename = 'pytests/config_functions/example_config'
    correct_dict = {'a': [12], 'b': ['mesh.vol'], 'c': [1]}
    correct_re_parse_dict = {'c': 'u+p'}
    new_variables = [{'u': 0, 'p': 1}]
    filetypes = ['.vol']

    return config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes


@pytest.fixture()
def presets_load_coefficientfunction_into_gridfunction() -> Tuple[Mesh, FESpace, FESpace, FESpace, Mesh, FESpace,
                                                                  FESpace, FESpace]:
    """
    Function to construct the presets for load_coefficientfunction_into_gridfunction.

    Returns:
        Tuple[Mesh, FESpace, FESpace, FESpace, Mesh, FESpace, FESpace, FESpace]:
            - mesh_2d: A 2D mesh.
            - fes_scalar_2d: A scalar finite element space for the 2D mesh.
            - fes_vector_2d: A vector finite element space for the 2D mesh.
            - fes_mixed_2d: A combined scalar and vector finite element space for the 2D mesh.
            - mesh_3d: A 3D mesh.
            - fes_scalar_3d: A scalar finite element space for the 3D mesh.
            - fes_vector_3d: A vector finite element space for the 3D mesh.
            - fes_mixed_3d: A combined scalar and vector finite element space for the 3D mesh.
    """
    # 2D
    mesh_2d = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    fes_scalar_2d = ngs.H1(mesh_2d, order=2)
    fes_vector_2d = ngs.HDiv(mesh_2d, order=2)
    fes_mixed_2d = ngs.FESpace([fes_vector_2d, fes_scalar_2d])

    # 3D
    geo_3d = CSGeometry()
    geo_3d.Add(OrthoBrick(Pnt(-1, 0, 0), Pnt(1, 1, 1)).bc('outer'))
    mesh_3d = ngs.Mesh(geo_3d.GenerateMesh(maxh=0.3))
    fes_scalar_3d = ngs.H1(mesh_3d, order=2)
    fes_vector_3d = ngs.HDiv(mesh_3d, order=2)
    fes_mixed_3d = ngs.FESpace([fes_vector_3d, fes_scalar_3d])

    return mesh_2d, fes_scalar_2d, fes_vector_2d, fes_mixed_2d, mesh_3d, fes_scalar_3d, fes_vector_3d, fes_mixed_3d


class TestParseStr:
    """ Class to test parse_to_str. """
    def test_1(self):
        """ Check that strings of the specified filetypes get kept as strings. """
        filetype = '.vol'
        input_str = 'mesh' + filetype
        output_str, variable_eval = parse_str(input_str, 'import_functions.py', None, [{}], [filetype])

        assert output_str == input_str # The input string should not have been parsed.
        assert not variable_eval       # The input string should not be flagged for re-parsing.

    def test_2(self):
        """
        Check that strings of a non-specified filetype are parsed. This should raise an error since they wouldn't be
        parseable.
        """
        filetype = '.vol'
        unspecified_filetype = '.stl'
        input_str = 'mesh' + unspecified_filetype

        with pytest.raises(pyparsing.ParseException): # Expect a parser error.
            parse_str(input_str, 'import_functions.py', None, [{}], [filetype])

    def test_3(self):
        """ Check that generic strings are parsed. """
        input_str = '3^2 + 6'            # Need to use a basic expression, can't directly compare coefficientfunctions.
        parsed_str = 3.0**2 + 6.0
        output_str, variable_eval = parse_str(input_str, 'import_functions.py', None)

        assert output_str == parsed_str  # The input string was parsed correctly.

    def test_4(self):
        """ Check that non-string inputs do not get parsed and just get passed through. """
        input_obj = [1, 2, 3]
        output_obj, variable_eval = parse_str(input_obj, 'import_functions.py', None)

        assert output_obj == input_obj  # The input object should not have been parsed.
        assert not variable_eval        # The input object should not be flagged for re-parsing.


class TestConvertStrToDict:
    """ Class to test convert_str_to_dict. """
    def test_1(self, load_configfile, presets_convert_str_to_dict):
        """ Check normal functioning. """
        config_section = 'TEST LOAD CONFIG'
        config_key = 'test_1'
        config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes = presets_convert_str_to_dict

        config = load_configfile(config_filename)
        input_str = config[config_section][config_key]
        output_dict, output_re_parse_dict = convert_str_to_dict(input_str, 'import_functions.py', [ngs.Parameter(0.0)],
                                                                None, new_variables, filetypes)

        assert output_dict == correct_dict                   # Check that the parameter dictionary is correct.
        assert output_re_parse_dict == correct_re_parse_dict # Check that the re-parse dictionary is correct.

    def test_2(self, load_configfile, presets_convert_str_to_dict):
        """ Check that newlines are necessary separators. """
        config_section = 'TEST LOAD CONFIG'
        config_key = 'test_2'
        config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes = presets_convert_str_to_dict

        config = load_configfile(config_filename)
        input_str = config[config_section][config_key]

        with pytest.raises(Exception):  # Expect an error.
            convert_str_to_dict(input_str, 'import_functions.py', [ngs.Parameter(0.0)], None, new_variables, filetypes)

    def test_3(self, load_configfile, presets_convert_str_to_dict):
        """ Check that -> is a necessary separator. """
        config_section = 'TEST LOAD CONFIG'
        config_key = 'test_3'
        config_filename, correct_dict, correct_re_parse_dict, new_variables, filetypes = presets_convert_str_to_dict

        config = load_configfile(config_filename)
        input_str = config[config_section][config_key]

        with pytest.raises(Exception):  # Expect an error.
            convert_str_to_dict(input_str, 'import_functions.py', [ngs.Parameter(0.0)], None, new_variables, filetypes)


class TestLoadCoefficientFunctionIntoGridFunction:
    """ Class to test load_coefficientfunction_into_gridfunction. """
    def test_1(self, presets_load_coefficientfunction_into_gridfunction):
        """ Check that a coefficientfunction can be loaded into a full gridfunction correctly. Check 2D and 3D. """
        mesh_2d, fes_scalar_2d, fes_vector_2d, fes_mixed_2d, mesh_3d, fes_scalar_3d, fes_vector_3d, fes_mixed_3d = presets_load_coefficientfunction_into_gridfunction

        gfu_scalar_2d = ngs.GridFunction(fes_scalar_2d)
        gfu_scalar_3d = ngs.GridFunction(fes_scalar_3d)
        cfu_scalar = ngs.CoefficientFunction(ngs.x + ngs.y)
        coef_dict = {None: cfu_scalar}

        # Check 2D
        load_coefficientfunction_into_gridfunction(gfu_scalar_2d, coef_dict)
        err_2d = ngs.Integrate((cfu_scalar - gfu_scalar_2d) * (cfu_scalar - gfu_scalar_2d), mesh_2d)
        assert math.isclose(err_2d, 0.0, abs_tol=1e-16)

        # Check 3D
        load_coefficientfunction_into_gridfunction(gfu_scalar_3d, coef_dict)
        err_3d = ngs.Integrate((cfu_scalar - gfu_scalar_3d) * (cfu_scalar - gfu_scalar_3d), mesh_3d)
        assert math.isclose(err_3d, 0.0, abs_tol=1e-16)

    def test_2(self, presets_load_coefficientfunction_into_gridfunction):
        """
        Check that coefficientfunctions can be loaded into different component of a gridfunction correctly. Check 2D and
        3D. Also check scalar and vector coefficientfunctions.
        """
        mesh_2d, fes_scalar_2d, fes_vector_2d, fes_mixed_2d, mesh_3d, fes_scalar_3d, fes_vector_3d, fes_mixed_3d = presets_load_coefficientfunction_into_gridfunction

        cfu_scalar = ngs.CoefficientFunction(ngs.x + ngs.y)

        gfu_mixed_2d = ngs.GridFunction(fes_mixed_2d)
        cfu_vector_2d = ngs.CoefficientFunction((ngs.x, ngs.y))
        coef_dict_2d = {0: cfu_vector_2d, 1: cfu_scalar}

        gfu_mixed_3d = ngs.GridFunction(fes_mixed_3d)
        cfu_vector_3d = ngs.CoefficientFunction((ngs.x, ngs.y, ngs.z))
        coef_dict_3d = {0: cfu_vector_3d, 1: cfu_scalar}

        # Check 2D
        load_coefficientfunction_into_gridfunction(gfu_mixed_2d, coef_dict_2d)
        err_scalar_2d = ngs.Integrate((cfu_scalar - gfu_mixed_2d.components[1]) * (cfu_scalar - gfu_mixed_2d.components[1]), mesh_2d)
        err_vector_2d = ngs.Integrate((cfu_vector_2d - gfu_mixed_2d.components[0]) * (cfu_vector_2d - gfu_mixed_2d.components[0]), mesh_2d)
        assert math.isclose(err_scalar_2d, 0.0, abs_tol=1e-16)
        assert math.isclose(err_vector_2d, 0.0, abs_tol=1e-16)

        # Check 3D
        load_coefficientfunction_into_gridfunction(gfu_mixed_3d, coef_dict_3d)
        err_scalar_3d = ngs.Integrate((cfu_scalar - gfu_mixed_3d.components[1]) * (cfu_scalar - gfu_mixed_3d.components[1]), mesh_3d)
        err_vector_3d = ngs.Integrate((cfu_vector_3d - gfu_mixed_3d.components[0]) * (cfu_vector_3d - gfu_mixed_3d.components[0]), mesh_3d)
        assert math.isclose(err_scalar_3d, 0.0, abs_tol=1e-16)
        assert math.isclose(err_vector_3d, 0.0, abs_tol=1e-16)

    def test_3(self, presets_load_coefficientfunction_into_gridfunction):
        """ Check that passing in a coef_dict with multiple keys, one of which is None raises an error. """
        mesh_2d, fes_scalar_2d, fes_vector_2d, fes_mixed_2d, mesh_3d, fes_scalar_3d, fes_vector_3d, fes_mixed_3d = presets_load_coefficientfunction_into_gridfunction

        gfu = ngs.GridFunction(fes_scalar_2d)
        cfu_1 = ngs.CoefficientFunction(ngs.x + ngs.y)
        cfu_2 = ngs.CoefficientFunction(ngs.x * ngs.y)
        coef_dict = {None: cfu_1, 0: cfu_2}

        with pytest.raises(AssertionError):
            load_coefficientfunction_into_gridfunction(gfu, coef_dict)

    def test_4(self, presets_load_coefficientfunction_into_gridfunction):
        """ Check that the coefficientfunction and gridfunction dimensions must agree. Check 2D and 3D. """
        mesh_2d, fes_scalar_2d, fes_vector_2d, fes_mixed_2d, mesh_3d, fes_scalar_3d, fes_vector_3d, fes_mixed_3d = presets_load_coefficientfunction_into_gridfunction

        cfu_scalar = ngs.CoefficientFunction(ngs.x + ngs.y)

        gfu_mixed_2d = ngs.GridFunction(fes_mixed_2d)
        cfu_vector_2d = ngs.CoefficientFunction((ngs.x, ngs.y))

        gfu_mixed_3d = ngs.GridFunction(fes_mixed_3d)
        cfu_vector_3d = ngs.CoefficientFunction((ngs.x, ngs.y, ngs.z))

        # Check 2D.
        with pytest.raises(netgen.libngpy._meshing.NgException):
            coef_dict = {1: cfu_vector_2d} # Vector coefficientfunction but scalar gridfunction.
            load_coefficientfunction_into_gridfunction(gfu_mixed_2d, coef_dict)

        with pytest.raises(netgen.libngpy._meshing.NgException):
            coef_dict = {0: cfu_scalar}  # Scalar coefficientfunction but vector gridfunction.
            load_coefficientfunction_into_gridfunction(gfu_mixed_2d, coef_dict)

        # Check 3D.
        with pytest.raises(netgen.libngpy._meshing.NgException):
            coef_dict = {1: cfu_vector_3d} # Vector coefficientfunction but scalar gridfunction.
            load_coefficientfunction_into_gridfunction(gfu_mixed_3d, coef_dict)

        with pytest.raises(netgen.libngpy._meshing.NgException):
            coef_dict = {0: cfu_scalar}  # Scalar coefficientfunction but vector gridfunction.
            load_coefficientfunction_into_gridfunction(gfu_mixed_3d, coef_dict)
