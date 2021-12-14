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

from .base_config_functions import ConfigFunctions
from .load_config import load_coefficientfunction_into_gridfunction
from ..helpers.io import update_gridfunction_from_files
import ngsolve as ngs
from ngsolve import Parameter, GridFunction, CoefficientFunction, Mesh
from typing import Dict, Union, List


class ICFunctions(ConfigFunctions):
    """
    Class to hold the initial condition functions.
    """

    def __init__(self, config_rel_path: str, import_dir: str, mesh: Mesh, t_param: List[Parameter] = [Parameter(0.0)],
                 new_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]] = [{}]) -> None:
        super().__init__(config_rel_path, import_dir, mesh, t_param)

        # Load the IC dict from the IC configfile.
        self.ic_dict, self.ic_re_parse_dict = self.config.get_three_level_dict(self.import_dir, mesh, self.t_param,
                                                                               new_variables)

    def set_initial_conditions(self, gfu_IC: GridFunction, mesh: Mesh, model_name: str,
                               model_components: Dict[str, int]) -> None:
        """
        Function to load the initial conditions from their configfile into a gridfunction.

        Args:
            mesh: The model's mesh.
            model_name: Sometimes several models will use the same IC configfile. This identifies which model's ICs
                should be loaded.
            gfu_IC: A gridfunction to hold the initial conditions.
            model_components: Maps between variable names and their component in the model's finite element space.
        """

        coef_dict = {}
        file_dict = {}
        for var, marker_dict in self.ic_dict[model_name].items():
            # Determine which component of the gridfunction the initial condition is for
            if var == 'all':
                component = None

            else:
                component = model_components[var]

            # Check if the initial condition is specified as a file, a coefficientfunction, or a dict of material
            # markers and coefficientfunctions.
            if len(marker_dict) == 1:
                # One single file or coefficientfunction.
                assert next(iter(marker_dict.keys())) == 'all'

                # Take the first value in the list since that should correspond to the start time when the initial
                # condition is being loaded.
                val = next(iter(marker_dict.values()))[0]

                if val is None:
                    # No initial condition needed so just keep the zeroed out gridfunction.
                    return

                if isinstance(val, str):
                    # Check that the file exists.
                    val = self._find_rel_path_for_file(val)

                    file_dict[component] = val
                else:
                    coef_dict[component] = val

            else:
                # Need to construct a coefficientfunction from a set of material markers.
                coef = ngs.CoefficientFunction([marker_dict[mat] for mat in mesh.GetMaterials()])
                coef_dict[component] = coef

        if coef_dict:
            load_coefficientfunction_into_gridfunction(gfu_IC, coef_dict)

        if file_dict:
            update_gridfunction_from_files(gfu_IC, file_dict)

    def update_initial_conditions(self, t_param: List[Parameter],
                                  updated_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]],
                                  mesh: Mesh) \
            -> None:
        """
        Function to update the initial conditions with new values of the model_variables.

        Args:
            t_param: List of parameters representing the current time and previous time steps.
            updated_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.
            mesh: Mesh used by the model
        """

        for k1, v1 in self.ic_re_parse_dict.items():
            for k2, v2 in self.ic_re_parse_dict[k1].items():
                self.ic_dict[k1][k2] = self.re_parse(self.ic_dict[k1][k2], self.ic_re_parse_dict[k1][k2], t_param, updated_variables, mesh)
