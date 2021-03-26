from .expanded_config_parser import ConfigParser
import ngsolve as ngs
from ngsolve import CoefficientFunction, GridFunction, Parameter
from os import path
from typing import Dict, Union
from .load_config import parse_str


class ConfigFunctions:
    """
    Class to hold any functions from the config files.
    """

    def __init__(self, config_rel_path: str, t_param: ngs.Parameter = ngs.Parameter(0.0),
                 new_variables: Dict[str, Union[float, ngs.CoefficientFunction, ngs.GridFunction]] = {}) -> None:
        """
        Initializer

        Args:
            config_rel_path: The filename, and relative path, for the config file for this controller
            t_param: Parameter representing the current time
            new_variables: Model variables
        """
        # Set the run directory for the config functions.
        idx = config_rel_path[::-1].index('/')
        self.run_dir = config_rel_path[:len(config_rel_path) - idx]

        # Load the config file.
        self.config = ConfigParser(config_rel_path)

        # Set the time parameter.
        self.t_param = t_param

    def _find_rel_path_for_file(self, file_name: str) -> str:
        """
        Function to check if a file exists, returning a relative path to it.

        Args:
            file_name: The name of the file

        Returns:
            val: The path to the file, relative to the run directory

        """
        # Check current working directory.
        if not path.isfile(file_name):
            # Check the specific run directory.
            if not path.isfile(self.run_dir + '/' + file_name):
                # Check the main run directory.
                if not path.isfile(self.run_dir + '/../' + file_name):
                    raise FileNotFoundError('The given file does not exist.')
                else:
                    rel_file_path = self.run_dir + '/../' + file_name
            else:
                rel_file_path = self.run_dir + '/' + file_name
        else:
            rel_file_path = file_name

        return rel_file_path

    def re_parse(self, param_dict: Dict[str, Union[str, float, CoefficientFunction, GridFunction]],
                 re_parse_dict: Dict[str, str], t_param: Parameter,
                 updated_variables: Dict[str, Union[str, float, CoefficientFunction, GridFunction]]) -> Dict:
        """
        Iterates through a parameter dictionary and re-parses any expressions containing model variables to use the
        updated values of those variables.

        Args:
            param_dict: The parameter dictionary to update.
            re_parse_dict: Dictionary containing only the parameters that need to be re-parsed and their string
                           expressions.
            updated_variables: Dictionary containing any model variables and their updated values.

        Returns:
            ~: The updated parameter dictionary.
        """

        for key, val in re_parse_dict.items():
            # Re-parse the string expression and use to replace the parameter value in dict.
            val, variable_eval = parse_str(val, t_param, updated_variables)
            param_dict[key] = val

        return param_dict


