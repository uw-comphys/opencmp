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

from ngsolve import Mesh, Parameter, CoefficientFunction, GridFunction
from typing import List, Optional, Dict, Tuple, Union, Any, Callable
from . import parse_arithmetic


def parse_str(string: str, import_dir: str, t_param: Optional[List[Parameter]],
              new_variables: List[Dict[str, Union[int, str, float, CoefficientFunction, GridFunction]]] = [{}],
              filetypes: List[str] = ['.vol', '.sol', '.vtk'], mesh: Optional[Mesh] = None) \
        -> Tuple[Union[str, float, CoefficientFunction, list], Union[str, bool, Callable]]:
    """
    Checks if a string appears to be a path to a file and if not parses the string into Python code.

    Args:
        string: The string.
        import_dir: The path to the main run directory containing the file from which to import any Python functions.
        t_param: List of parameters representing the current time and previous time steps. If None, the parsed values
            have no possible time dependence and one single value is returned instead of a list of values corresponding
            to the parsed value at each time step.
        new_variables: List of dictionaries containing any new model variables and their values at each time step used
            in the time discretization scheme.
        filetypes: List of possible filetypes to consider.
        mesh: Mesh used by the model.

    Returns:
        Tuple[List[Union[str, float, CoefficientFunction]], Union[str, bool]]:
            - parsed_str: List containing the value of the parsed string at every time, or just the single value of the
              parsed string if it has no possible time dependence.
            - variable_eval: Whether or not the expression contains any of the new model variables (would need to be
              re-parsed if their values change).
    """

    if not isinstance(string, str):
        # Likely got a default value from config_defaults.
        if t_param is None:
            return string, False
        else:
            return [string] * len(t_param), False

    for item in filetypes:
        if string.endswith(item):
            # Filename, don't try to parse.
            if t_param is None:
                return string, False
            else:
                return [string] * len(t_param), False

    # Otherwise parse into Python code.
    # This needs to be done separately for each time step so the value of the parsed string includes the correct time at
    # each time step. However, variable_eval will remain the same for each time step since it has nothing to do with the
    # time value.
    if t_param is None:
        parsed_str, variable_eval = parse_arithmetic.eval_python(string, import_dir, mesh, new_variables, t_param, None)
    else:
        parsed_str = []
        for i in range(len(t_param)):
            tmp_parsed_str, variable_eval = parse_arithmetic.eval_python(string, import_dir, mesh, new_variables,
                                                                         t_param, i)
            parsed_str.append(tmp_parsed_str)

    return parsed_str, variable_eval


def convert_str_to_dict(string: str, import_dir: str, t_param: Optional[List[Parameter]], mesh: Mesh,
                        new_variables: List[Dict[str, Union[int, str, float, CoefficientFunction, GridFunction]]] = [{}],
                        filetypes: List[str] = ['.vol', '.sol', '.vtk'], all_str: bool = False) -> Tuple[Dict, Dict]:
    """
    Function to convert a string into a dict. The values of the dict may be parsed into Python
    code or left as strings.

    Args:
        string: The string.
        import_dir: The path to the main run directory containing the file from which to import any Python functions.
        t_param: List of parameters representing the current time and previous time steps. If None, the parsed values
            have no possible time dependence and one single value is returned instead of a list of values corresponding
            to the parsed value at each time step.
        mesh: Mesh used for the model
        new_variables: List of dictionaries containing any new model variables and their values at each time step used
            in the time discretization scheme.
        filetypes: List of possible filetypes to consider.
        all_str: If True, don't bother parsing any values, just load them as strings.

    Returns:
        Tuple[Dict, Dict]:
            - param_dict: Dictionary of the parameters from the string.
            - re_parse_dict: Dictionary containing only parameter values that may need to be re-parsed in the future.
    """

    # Replace extraneous whitespace characters.
    param_tmp = string.replace('\t', '')
    param_tmp = param_tmp.replace(' ', '')

    # If param_tmp contains multiple values, split them out into a list
    if '\n' in param_tmp:
        param_tmp_lst = param_tmp.split('\n')
        # Remove any empty strings resulting from extraneous newlines.
        param_tmp_lst = [item for item in param_tmp_lst if item != '']
    else:
        param_tmp_lst = [param_tmp]

    # Split by the delimiting character and stick the results in a dictionary.
    param_dict = {}
    re_parse_dict = {}
    val: Union[List[Any], Any] # Just for type-hinting.
    variable_eval: Union[str, bool, Callable] # Just for type-hinting.
    for item in param_tmp_lst:
        key, val_tmp = item.split('->')
        if all_str:
            # Don't bother parsing if the values should all be kept as strings.
            if t_param is None:
                val = val_tmp
            else:
                val = [val_tmp] * len(t_param)
            variable_eval = False
        else:
            val, variable_eval = parse_str(val_tmp, import_dir, t_param, new_variables, filetypes, mesh)
        param_dict[key] = val

        # If variable_eval is a string expression or imported Python function add it to re_parse_dict in case the
        # expression needs to be re-parsed with new variable values in the future.
        if isinstance(variable_eval, str) or callable(variable_eval):
            re_parse_dict[key] = variable_eval

    return param_dict, re_parse_dict


def load_coefficientfunction_into_gridfunction(gfu: GridFunction, coef_dict: Dict[Optional[int], CoefficientFunction]) \
        -> None:
    """
    Function to load a coefficientfunction(s) into a gridfunction.

    The coefficientfunction(s) may have a different dimension than the gridfunction and need to be loaded into a
    specific component of it.

    Args:
        coef_dict: Dictionary containing the coefficientfunction(s) and which component they belong to (keys).
        gfu: The gridfunction to load the values into.

    """

    for key, val in coef_dict.items():
        if key is None:
            # Confirm that coef_dict only has one value, otherwise the gfu values will be overwritten multiple times
            assert len(coef_dict) == 1
            gfu.Set(val)
        else:
            gfu.components[key].Set(val)
