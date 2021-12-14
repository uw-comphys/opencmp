from ngsolve.comp import GridFunction

from .base_config_functions import ConfigFunctions
from ngsolve import CoefficientFunction, Mesh, Parameter
from typing import Any, Dict, List, Tuple, Union, cast


class ControllerFunctions(ConfigFunctions):
    """
    Class to process values for the controller
    """

    def __init__(self, config_rel_path: str, import_dir: str, mesh: Mesh, t_param: List[Parameter],
                 new_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]] = [{}]) -> None:
        super().__init__(config_rel_path, import_dir, mesh, t_param, new_variables)

        tmp_vc_dict = _convert_dict_entries_to_list(self.config.get_one_level_dict('CONTROL_VARIABLES',
                                                                                   import_dir, mesh, self.t_param)[0])
        tmp_mv_dict = _convert_dict_entries_to_list(self.config.get_one_level_dict('MANIPULATED_VARIABLES',
                                                                                   import_dir, mesh, self.t_param)[0])

        self._control_var_dict: Dict[str, Union[List[str], List[float], List[Tuple[float]]]]    = tmp_vc_dict
        self._manipulated_var_dict: Dict[str, List[str]]                                        = tmp_mv_dict

    def get_control_variables(self) -> List[Tuple[str, Tuple[float, ...], Union[float, Tuple[float, ...]], float]]:
        """
        Function to get the control variables in a structured format.

        Returns:
            ~: The control variables as a set of nested lists [[name_1, pos_1, val_1, index_1], [...], ...]
        """

        # Get variables
        variable_names     = cast(List[str],                self._control_var_dict['variable_names'])
        location_positions = cast(List[Tuple[float, ...]],  self._control_var_dict['location_positions'])
        variable_index     = cast(List[float],              self._control_var_dict['index'])

        values: List[Tuple[float, ...]] = []

        for val in self._control_var_dict['values']:
            if type(val) is float:
                values.append(tuple([cast(float, val)]))
            elif type(val is Tuple):
                if type(val[0]) is float:
                    values.append(cast(Tuple[float], val))
                else:
                    raise ValueError('Wrong variable type in values for controller')
            else:
                raise ValueError('Wrong variable type in values for controller')

        # Zip together and return
        return [(name, pos, val, index)
                for name, pos, val, index
                in zip(variable_names, location_positions, values, variable_index)]

    def get_manipulated_variables(self) -> List[Tuple[str, str, str]]:
        """
        Function to get the manipulated variables in a structured format.

        Returns:
            ~: The manipulated variables as a set of nested lists [[type_1, var_1, loc_1], [...], ...]
        """

        # Get variables
        types: List[str]          = self._manipulated_var_dict['types']
        variable_names: List[str] = self._manipulated_var_dict['variable_names']
        location_names: List[str] = self._manipulated_var_dict['location_names']

        # Zip together and return
        return [(_type, var, loc) for _type, var, loc in zip(types, variable_names, location_names)]


def _convert_dict_entries_to_list(_dict: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Function to convert the top level entries in a dictionary into a list.

    Args:
        _dict: The dictionary to work on

    Return:
        ~: The dictionary after all top level entries have been converted to lists
    """
    dict_out = {}

    for key in _dict:
        if type(_dict[key]) is list:
            dict_out[key] = _dict[key]
        else:
            dict_out[key] = [_dict[key]]

    return dict_out
