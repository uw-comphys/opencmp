import ngsolve as ngs
from typing import List, Optional, Dict, Tuple
from . import parse_arithmetic


def parse_str(string: str, t_param: ngs.Parameter, new_variables: Dict[str, Optional[int]] = {},
              filetypes: List[str] = ['.vol', '.sol', '.vtk']) -> Dict:
    """
    Checks if a string appears to be a path to a file and if not parses the string into Python code.
    Args:
        string: The string.
        t_param: Time parameter used within the Python code.
        new_variables: Dictionary containing any new model variables and their values.
        filetypes: List of possible filetypes to consider.

    Returns:
        parsed_str: The parsed string.
        variable_eval: Whether or not the expression contains any of the new model variables (would need to be re-parsed
                       if their values change).
    """

    if not isinstance(string, str):
        # Likely got a default value from config_defaults.
        return string, False

    for item in filetypes:
        if string.endswith(item):
            # Filename, don't try to parse.
            return string, False

    # Otherwise parse into Python code.
    parsed_str, variable_eval = parse_arithmetic.eval_python(string, t_param, new_variables)

    return parsed_str, variable_eval


def convert_str_to_dict(string: str, t_param: ngs.Parameter, new_variables: Dict[str, Optional[int]] = {},
                        filetypes: List[str] = ['.vol', '.sol', '.vtk']) -> Tuple[Dict, Dict]:
    """
    Function to convert a string into a dict. The values of the dict may be parsed into Python
    code or left as strings.

    Args:
        string: The string.
        t_param: Time parameter used within the Python code.
        new_variables: Dictionary containing any new model variables and their values.
        filetypes: List of possible filetypes to consider.

    Returns:
        param_dict: Dict of the parameters from the string.
        re_parse_dict: Dict containing only parameter values that may need to be re-parsed in the future.
    """

    # Replace extraneous whitespace characters.
    param_tmp = string.replace('\t', '')
    param_tmp = param_tmp.replace(' ', '')

    # See if param_tmp has multiple values.
    try:
        param_tmp = param_tmp.split('\n')
        # Remove any empty strings resulting from extraneous newlines.
        param_tmp = [item for item in param_tmp if item != '']
    except:
        pass

    # Split by the delimiting character and stick the results in a dictionary.
    param_dict = {}
    re_parse_dict = {}
    for item in param_tmp:
        key, val_tmp = item.split('->')
        val, variable_eval = parse_str(val_tmp, t_param, new_variables, filetypes)
        param_dict[key] = val

        # If variable_eval is a string expression add it to re_parse_dict in case the expression needs to be re-parsed
        # with new variable values in the future.
        if isinstance(variable_eval, str):
            re_parse_dict[key] = variable_eval

    return param_dict, re_parse_dict


def load_coefficientfunction_into_gridfunction(gfu: ngs.GridFunction,
                                               coef_dict: Dict[Optional[int], ngs.CoefficientFunction]) -> None:
    """
    Function to load a coefficientfunction(s) into a gridfunction. Coefficientfunction may have a different dimension
    than the gridfunction and need to be loaded into a specific component.

    Args:
        coef_dict: Dict containing the coefficientfunction(s) and which component they belong to (keys)
        gfu: The gridfunction to load the values into

    """

    for key, val in coef_dict.items():
        if key is None:
            # Confirm that coef_dict only has one value, otherwise the gfu values will be overwritten multiple times
            assert len(coef_dict) == 1
            gfu.Set(val)
        else:
            gfu.components[key].Set(val)
