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

from .expanded_config_parser import ConfigParser
from ngsolve import CoefficientFunction, GridFunction, Mesh, Parameter
from os import path
from typing import Dict, Union, Optional, List
from .load_config import parse_str


class ConfigFunctions:
    """
    Class to hold any functions from the config files.
    """

    def __init__(self, config_rel_path: str, import_dir: str, mesh: Mesh, t_param: List[Parameter],
                 new_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]] = [{}]) -> None:
        """
        Initializer

        Args:
            config_rel_path: The filename, and relative path, for the config file for this controller.
            import_dir: The path to the main run directory containing the file from which to import any Python functions.
            t_param: List of parameters representing the current time and previous time steps.
            new_variables: List of dictionaries containing any new model variables and their values at each time step
                used in the time discretization scheme.
        """
        # Set the run directory for the config functions.
        # Files could get accessed at run_dir + '/' + <path>. Make sure that if the config file is in the current
        # directory run_dir is specified such that this will still work.
        if '/' in config_rel_path:
            idx = config_rel_path[::-1].index('/')
            self.run_dir = config_rel_path[:len(config_rel_path) - idx - 1]
        else:
            self.run_dir = '.'

        # Load the config file.
        self.config = ConfigParser(config_rel_path)

        # Check if the config parser is empty. If it is that might be because the file doesn't exist, in which case
        # raise an error. Alternatively it might just mean that the file is empty because it doesn't need to be used.
        if self.config == []:
            if not path.exists(config_rel_path):
                raise FileNotFoundError('File {} does not exist.'.format(config_rel_path))

        # Set the time parameter.
        self.t_param = t_param

        # Set the import file path.
        self.import_dir = import_dir

    def _find_rel_path_for_file(self, file_name: str) -> str:
        """
        Function to check if a file exists, returning a relative path to it.

        Args:
            file_name: The name of the file.

        Returns:
            The path to the file, relative to the run directory.

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

    def re_parse(self, param_dict: Dict[str, Union[str, float, CoefficientFunction, GridFunction, list]],
                 re_parse_dict: Dict[str, str], t_param: Optional[List[Parameter]],
                 updated_variables: List[Dict[str, Union[int, str, float, CoefficientFunction, GridFunction]]],
                 mesh: Mesh) -> Dict:
        """
        Iterates through a parameter dictionary and re-parses any expressions containing model variables to use the
        updated values of those variables.

        Args:
            param_dict: The parameter dictionary to update.
            re_parse_dict: Dictionary containing only the parameters that need to be re-parsed and their string
                expressions.
            t_param: List of parameters representing the current time and previous time steps. If None, the re-parsed
                values have no possible time dependence and one single value is returned instead of a list of values
                corresponding to the re-parsed value at each time step.
            mesh: Mesh used by the model
            updated_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.

        Returns:
            The updated parameter dictionary.
        """

        for key, val in re_parse_dict.items():
            if callable(val):
                # Need t_param to actually be a list of time parameters to be able to call an imported Python function.
                assert t_param is not None

                # The expression is an imported Python function, so just re-evaluate it with the new time and model
                # variable values.
                param_dict[key] = [val(t_param, updated_variables, mesh, i) for i in range(len(t_param))]
            else:
                # Re-parse the string expression and use to replace the parameter value in dict.
                re_parse_val, variable_eval = parse_str(val, self.import_dir, t_param, updated_variables, mesh=mesh)
                param_dict[key] = re_parse_val

        return param_dict


