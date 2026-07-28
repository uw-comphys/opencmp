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
import ngsolve as ngs
from ngsolve import Mesh, Parameter, CoefficientFunction, GridFunction, FESpace
from typing import Dict, Union, List


class RefSolFunctions(ConfigFunctions):
    """
    Class to hold the reference solutions.
    """

    def __init__(self, config_rel_path: str, import_dir: str, mesh: Mesh, t_param: List[Parameter] = [Parameter(0.0)],
                 new_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]] = [{}]) -> None:
        super().__init__(config_rel_path, import_dir, mesh, t_param)

        # Load the reference solution dict from the reference solution configfile.
        try:
            ref_sols, ref_sols_re_parse = self.config.get_one_level_dict('REFERENCE SOLUTIONS', self.import_dir, mesh,
                                                                         self.t_param, new_variables)
        except KeyError:
            # No options specified for L2_error.
            ref_sols = {}
            ref_sols_re_parse = {}

        try:
            # Note that all of the metrics are time-independent since they are just names of error metrics.
            metrics, metrics_re_parse = self.config.get_one_level_dict('METRICS', self.import_dir, mesh,
                                                                       new_variables=new_variables, all_str=True)
        except KeyError:
            # No options specified for other metrics.
            metrics = {}
            metrics_re_parse = {}

        self.ref_sol_dict = {'ref_sols': ref_sols, 'metrics': metrics}
        self.ref_sol_re_parse_dict = {'ref_sols': ref_sols_re_parse, 'metrics': metrics_re_parse}

    def set_ref_solution(self, fes: FESpace, model_components: Dict[str, int]):
        """
        Function to load the reference solutions from their configfile into a dictionary, including loading any saved
        gridfunctions.

        Args:
            fes: The finite element space of the model.
            model_components: Maps between variable names and their component in the finite element space.

        Returns:
            Dictionary of reference solutions.
        """

        # Turn 'all' into a separate reference solution for each variable. Assuming that anytime 'all' is used the
        # reference solution is being loaded in from file since there is no way to parse closed form expressions for
        # multiple gridfunction components.
        if 'all' in self.ref_sol_dict['ref_sols'].keys():
            val = self.ref_sol_dict['ref_sols'].pop('all')
            for var in model_components.keys():
                self.ref_sol_dict['ref_sols'][var] = val

        for var, val_lst in self.ref_sol_dict['ref_sols'].items():
            for i in range(len(val_lst)):
                val = val_lst[i]

                if isinstance(val, str):
                    # Need to load a gridfunction from file.

                    # Check that the file exists.
                    val = self._find_rel_path_for_file(val)

                    component = model_components[var]

                    # Use a component of the finite element space.
                    gfu_val = ngs.GridFunction(fes.components[component])
                    gfu_val.Load(val)

                    # Replace the value in the ref_sol dictionary.
                    self.ref_sol_dict['ref_sols'][var][i] = gfu_val

            else:
                # Already parsed
                pass

        return self.ref_sol_dict

    def update_ref_solutions(self, t_param: List[Parameter],
                             updated_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]],
                             mesh: Mesh) \
            -> None:
        """
        Function to update the reference solutions with new values of the model_variables.

        Args:
            t_param: List of parameters representing the current time and previous time steps.
            updated_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.
            mesh: Mesh used by the model
        """

        for k1, v1 in self.ref_sol_re_parse_dict.items():
            self.ref_sol_dict[k1] = self.re_parse(self.ref_sol_dict[k1], self.ref_sol_re_parse_dict[k1], t_param, updated_variables, mesh)

