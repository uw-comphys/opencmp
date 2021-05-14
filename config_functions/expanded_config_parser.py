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

import configparser
from .load_config import parse_str, convert_str_to_dict
from os.path import isfile
from typing import Any, Dict, List, Type, TypeVar, Union, cast, Optional, Tuple
from ngsolve import CoefficientFunction, Parameter, GridFunction

T = TypeVar('T')

config_defaults: Dict = {
    'MESH': {'filename': 'REQUIRED',
             'curved_elements': False},
    'FINITE ELEMENT SPACE': {'elements': 'REQUIRED',
                             'interpolant_order': 'REQUIRED',
                             'no_constrained_dofs': False},
    'DG': {'DG': False,
           'interior_penalty_coefficient': 10.0},
    'SOLVER': {'solver': 'default',
               'preconditioner': 'default',
               'solver_tolerance': 1e-8,
               'solver_max_iterations': 100,
               'linearization_method': 'Oseen',
               'nonlinear_tolerance': {'absolute': 0.0, 'relative': 'REQUIRED'},
               'nonlinear_max_iterations': 10},
    'TRANSIENT': {'transient': False,
                  'scheme': 'implicit euler',
                  'time_range': [0.0, 5.0],
                  'dt': 0.001,
                  'dt_tolerance': {'absolute': 0.0, 'relative': 'REQUIRED'},
                  'dt_range': [1e-12, 0.1],
                  'maximum_rejected_solves': 1000},
    'ERROR ANALYSIS': {'check_error': False,
                       'check_error_every_timestep': False,
                       'convergence_test': {'h': False, 'p': False},
                       'error_average': [],
                       'num_refinements': 4},
    'OTHER': {'num_threads': 4,
              'messaging_level': 0,
              'model': 'REQUIRED',
              'component_names': [],
              'run_dir': 'REQUIRED'},
    'VISUALIZATION': {'save_to_file': False,
                      'save_type': '.sol',
                      'save_frequency': ['1', 'numit'],
                      'subdivision': -1},
    'DIM': {'diffuse_interface_method': False,
            'dim_dir': 'REQUIRED',
            'mesh_dimension': 2,
            'num_mesh_elements': {'x': 59, 'y': 59, 'z': 59},
            'num_phi_mesh_elements': {},
            'mesh_scale': {'x': 1.0, 'y': 1.0, 'z': 1.0},
            'mesh_offset': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'interface_width_parameter': 1e-5,
            'mnum': 1.0,
            'close': False,
            'quad_mesh': True},
    'PHASE FIELDS': {'load_method': 'REQUIRED',
                     'invert_phi': False,
                     'stl_filename': 'REQUIRED',
                     'phase_field_filename': {'phi': 'REQUIRED', 'grad_phi': None, 'mag_grad_phi': None},
                     'save_to_file': True},
    'DIM BOUNDARY CONDITIONS': {'multiple_bcs': False,
                                'overlap_interface_parameter': -1,
                                'remainder': False},
    'RIGID BODY MOTION': {'rotation_speed': 'REQUIRED',
                          'center_of_rotation': 'REQUIRED',
                          'translation_vector': 'REQUIRED'}
}


class ConfigParser(configparser.ConfigParser):
    """
    A ConfigParser extended to have several useful functions added to it.
    """
    def __init__(self, config_file_path: str) -> None:
        super().__init__()

        # Check that the file exists.
        if not isfile(config_file_path):
            raise FileNotFoundError('The given config file \"{}\" does not exist.'.format(config_file_path))

        self.read(config_file_path)

    def get_one_level_dict(self, config_section: str, t_param: Optional[List[Parameter]] = None,
                           new_variables: List[Dict[str, Optional[int]]] = [{}], all_str: bool = False) \
            -> Tuple[Dict, Dict]:
        """
        Function to load parameters from a config file into a single-level dictionary.

        | The values in the lowest level dictionary are either parsed into Python code or kept as strings if
        | they represent paths to .sol files. All keys are kept as strings.
        |
        | Ex: model_params_dict = {'density': float,
        |                          'viscosity': float,
        |                          'source': CoefficientFunction}

        Args:
            config_section: The section of the config file to load parameters from.
            t_param: List of parameters representing the current time and previous time steps. If None, the re-parsed
                values have no possible time dependence and one single value is returned instead of a list of values
                corresponding to the re-parsed value at each time step.
            new_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.
            all_str: If True, don't bother parsing any values, just load them as strings.

        Returns:
            Tuple[Dict, Dict]
                - dict_one: Dictionary containing the config file values.
                - re_parse_dict: Dictionary containing only parameter values that may need to be re-parsed in the
                    future.
        """

        dict_one: Dict[str, Union[str, float, CoefficientFunction, list]] = {}
        re_parse_dict: Dict[str, str] = {}
        for key in self[config_section]:
            if all_str:
                # If all_str option is passed none of the parameters should ever need to be re-parsed.
                val_str = self.get_list([config_section, key], str)

                if t_param is None:
                    dict_one[key] = val_str
                else:
                    dict_one[key] = [val_str for _ in t_param]
            else:
                val_str = self.load_param_simple([config_section, key])
                val, variable_eval = parse_str(val_str, t_param, new_variables, ['.sol'])
                dict_one[key] = val

                if isinstance(variable_eval, str):
                    # Add the string expression to re_parse_dict in case it needs to be re-parsed in the future,
                    re_parse_dict[key] = variable_eval

        return dict_one, re_parse_dict

    def get_two_level_dict(self, config_section: str, t_param: Optional[List[Parameter]] = None,
                           new_variables: List[Dict[str, Optional[int]]] = [{}]) -> Tuple[Dict, Dict]:
        """
        Function to load parameters from a config file into a two-level dictionary. 
        
        | The values in the lowest level dictionary are either parsed into Python code or
        | kept as strings if they represent paths to .sol files. All other keys and values
        | are kept as strings.
        |
        | Ex: model_functions_dict = {'source': {'c1': CoefficientFunction,
        |                                        'c2': CoefficientFunction}
        |                             }

        Args:
            config_section: The section of the config file to load parameters from.
            t_param: List of parameters representing the current time and previous time steps. If None, the re-parsed
                values have no possible time dependence and one single value is returned instead of a list of values
                corresponding to the re-parsed value at each time step.
            new_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.

        Returns:
            Tuple[Dict, Dict]
                - dict_one: Dictionary containing the config file values.
                - re_parse_dict: Dictionary containing only parameter values that may need to be re-parsed in the
                    future.
        """

        # Top level dictionaries.
        dict_one: Dict[str, Dict[str, Union[str, float, CoefficientFunction, list]]] = {}
        re_parse_dict: Dict[str, Dict[str, str]] = {}

        for key in self[config_section]:
            # 2nd level dictionary
            dict_two, re_parse_dict_two = convert_str_to_dict(self[config_section][key], t_param, new_variables, ['.sol'])
            dict_one[key] = dict_two
            re_parse_dict[key] = re_parse_dict_two

        return dict_one, re_parse_dict

    def get_three_level_dict(self, t_param: Optional[List[Parameter]] = None,
                             new_variables: List[Dict[str, Optional[int]]] = [{}], ignore=[]) -> Tuple[Dict, Dict]:
        """
        Function to load parameters from a config file into a three-level dictionary.

        | The values in the lowest level dictionary are either parsed into Python code or kept as strings if they
        | represent paths to .sol files. All other keys and values are kept as strings.
        |
        | Ex: bc_dict = {'dirichlet': {'u': {marker1: CoefficientFunction,
        |                                    marker2: CoefficientFunction},
        |                              'p': {marker3: CoefficientFunction}
        |                              },
        |                'neumann':   {'p': {marker4: CoefficientFunction}
        |                              }
        |                }

        Args:
            t_param: List of parameters representing the current time and previous time steps. If None, the re-parsed
                values have no possible time dependence and one single value is returned instead of a list of values
                corresponding to the re-parsed value at each time step.
            new_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.
            ignore: List of section headers to ignore if only part of the config file should be read.

        Returns:
            Tuple[Dict, Dict]:
                - dict_one: Dictionary containing the config file values.
                - re_parse_dict: Dictionary containing only parameter values that may need to be re-parsed in the
                    future.
        """

        # Keys for the top level dictionaries
        keys_one = [item for item in self.sections() if item not in ignore]

        # Top level dictionaries
        dict_one: Dict[str, Dict[str, Dict[str, Union[str, float, CoefficientFunction, list]]]] = {}
        re_parse_dict: Dict[str, Dict[str, Dict[str, str]]] = {}

        for k1 in keys_one:
            # 2nd level dictionaries
            dict_two = {}
            re_parse_dict_two = {}
            for k2 in self[k1]:
                # 3rd level dictionaries
                dict_three, re_parse_dict_three = convert_str_to_dict(self[k1][k2], t_param, new_variables, ['.sol'])
                dict_two[k2] = dict_three
                re_parse_dict_two[k2] = re_parse_dict_three
            dict_one[k1.lower()] = dict_two
            re_parse_dict[k1.lower()] = re_parse_dict_two

        return dict_one, re_parse_dict

    def load_param_simple(self, config_keys: List[str], quiet: bool = False) -> str:
        """
        Loads a parameter specified in the given config file. Keeps the parameter as a string and does not try to split
        it into a list of values.

        Args:
            config_keys: The keys that specify the location of the parameter in the config file (header, subheader...).
            quiet: If True suppresses the warning about the default value being used for a parameter, and about not
                converting a number to a string.

        Returns:
            The parameter value.
        """

        section, key = config_keys

        try:
            param = self[section][key]
        except:
            try:
                param = config_defaults.get(section).get(key)
            except:
                raise ValueError('{0}, {1} is not a valid parameter.'.format(section, key))

            if param == 'REQUIRED':
                raise ValueError('No default available for {0}, {1}. '
                                 'Please specify a value in the config file.'.format(section, key))
            else:
                if not quiet:
                    print('Using the default value of {0} for {1}, {2}.'.format(param, section, key))

        return param

    def _load_param(self, config_keys: List[str], val_type: Type[T], quiet: bool = False) -> List[T]:
        """
        Loads a parameter specified in the given config file. Can also loads lists of the specified parameter type.

        Args:
            config_keys: The keys that specify the location of the parameter in the config file (header, subheader...).
            val_type: The expected parameter type.
            quiet: If True suppresses the warning about the default value being used for a parameter, and about not
                converting a number to a string.

        Returns:
            The parameter value or a list of the parameter's values.
        """

        section, key = config_keys
        param_tmp: Any

        try:
            param_tmp = self[section][key].split(', ')
        except:
            try:
                param_tmp = config_defaults.get(section).get(key)
            except:
                raise ValueError('{0}, {1} is not a valid parameter.'.format(section, key))

            if param_tmp == 'REQUIRED':
                raise ValueError('No default available for {0}, {1}. '
                                 'Please specify a value in the config file.'.format(section, key))
            else:
                if not quiet:
                    print('Using the default value of {0} for {1}, {2}.'.format(param_tmp, section, key))

        try:
            param: List[T] = []

            if type(param_tmp) is not list:
                param_tmp = [param_tmp]

            for item in param_tmp:
                # Check to make sure that a number is not accidentally converted to something else.
                if not quiet:
                    if not (val_type is int or val_type is float):
                        try:
                            float(item)
                            print('You are trying to convert ' + item + ' to type ' + val_type.__name__ +
                                  ' but it appears to be a number. Is this a mistake?')
                        except:
                            # If it can't be converted to a number, do nothing
                            pass

                if val_type is bool:
                    if isinstance(item, str):
                        param.append(item == 'True')
                    elif isinstance(item, bool):
                        param.append(item)
                else:
                    param.append(val_type(item))
        except:
            raise TypeError('Incorrect value type for {} in {}.'.format(key, section))

        return param

    def get_dict(self, config_keys: List[str], t_param: Optional[List[Parameter]] = None, quiet: bool = False,
                 all_str: bool = False) -> Dict[str, Any]:
        """
        Function to load a one level dictionary of parameters from the config file.

        Use instead of get_one_level_dict if the dictionary is denoted by a "->" separator.

        Args:
            config_keys: The keys needed to access the parameters from the config file.
            t_param: List of parameters representing the current time and previous time steps. If None, the re-parsed
                values have no possible time dependence and one single value is returned instead of a list of values
                corresponding to the re-parsed value at each time step.
            quiet: If True suppresses the warning about the default value being used for a parameter.
            all_str: If True, don't bother parsing any values, just load them as strings.

        Returns:
            Dictionary of the parsed parameters from the config file.
        """
        string = self.load_param_simple(config_keys, quiet)

        if isinstance(string, str):
            # If loaded from the config file the string will need to be parsed into a dict, otherwise if it was taken
            # from config_defaults it will already be a dict.
            param_dict, re_parse_dict = convert_str_to_dict(string, t_param, all_str=all_str)
        elif isinstance(string, dict):
            param_dict = string
            re_parse_dict = {}
        else:
            raise TypeError('Need to update config_defaults. {0}, {1} should be a dictionary.'
                            .format(config_keys[0], config_keys[1]))

        if re_parse_dict:
            # Assuming that any parameters obtained with get_dict are not functions of the model variables and will
            # never need to be re-parsed.
            raise NotImplementedError('No way to re-parse this parameter. Should be added to a ConfigFunctions object so'
                                      ' re-parsing can be tracked.')

        # Check that param_dict has all the expected keys. If one or more keys are missing, replace them with their
        # default values (or raise an error that that key's value must be specified).
        default = config_defaults.get(config_keys[0], {}).get(config_keys[1], {})
        if default == 'REQUIRED':
            # No defaults available so just assume that what the user gave is correct.
            pass
        elif isinstance(default, dict):
            for key, val in default.items():
                if key not in param_dict.keys():
                    # Check that param_dict isn't missing any required items.
                    if val == 'REQUIRED':
                        raise ValueError('No default available for {0}, {1}, {2}. Please specify a value in the config '
                                         'file.'.format(config_keys[0], config_keys[1], key))
                    else:
                        # If there is a default value for the item add it into param_dict.
                        if t_param is None:
                            param_dict[key] = val
                        else:
                            param_dict[key] = [val for _ in t_param]
                elif param_dict[key] == 'REQUIRED':
                    # Check that any default values assigned to param_dict are real and not values that are required to
                    # be specified.
                    raise ValueError('No default available for {0}, {1}, {2}. Please specify a value in the config '
                                     'file.'.format(config_keys[0], config_keys[1], key))
        else:
            raise TypeError('Need to update config_defaults. {0}, {1} should be a dictionary.'
                            .format(config_keys[0], config_keys[1]))

        return param_dict

    def get_list(self, config_keys: List[str], val_type: Type[T], quiet: bool = False) -> List[T]:
        """
        Function to load a list of parameters from the config file.

        Args:
            config_keys: The keys needed to access the parameters from the config file.
            val_type: The type that each parameter is supposed to be.
            quiet: If True suppresses the warning about the default value being used for a parameter.

        Returns:
            List of the parameters from the config file in the type specified.
        """
        ret = self._load_param(config_keys, val_type, quiet)

        # Ensure that it's a list
        assert type(ret) == list

        return cast(List[val_type], ret)

    def get_item(self, config_keys: List[str], val_type: Type[T], quiet: bool = False) -> T:
        """
        Function to load a parameter from the config file.

        Args:
            config_keys: The keys needed to access the parameter from the config file.
            val_type: The type that the parameter is supposed to be.
            quiet: If True suppresses the warning about the default value being used for a parameter.

        Returns:
            The parameter from the config file in the type specified.
        """
        ret = self._load_param(config_keys, val_type, quiet)

        # Ensure that it's a single value
        assert len(ret) == 1 and type(ret[0]) == val_type

        ret_value = ret[0]

        return cast(val_type, ret_value)
