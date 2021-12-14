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
from ngsolve import Parameter, CoefficientFunction, GridFunction, FESpace, Mesh
from typing import Dict, Tuple, Union, List, Optional


class BCFunctions(ConfigFunctions):
    """
    Class to hold the boundary condition functions.
    """

    def __init__(self, config_rel_path: str, import_dir: str, mesh: Mesh, t_param: List[Parameter] = [Parameter(0.0)],
                 new_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]] = [{}]) -> None:
        super().__init__(config_rel_path, import_dir, mesh, t_param)

        # Load the BC dict from the BC configfile.
        self.bc_dict, self.bc_re_parse_dict = self.config.get_three_level_dict(self.import_dir, None, self.t_param,
                                                                               new_variables,
                                                                               ignore=['VERTICES', 'CENTROIDS'])

    def load_bc_gridfunctions(self, bc_dict: Dict, fes: FESpace, model_components: Dict[str, Optional[int]]) -> Dict:
        """
        Function to load any saved gridfunctions that should be used to specify BCs.

        Args:
            bc_dict: Dictionary specifying the BCs and their mesh markers for each variable.
            fes: The finite element space of the model.
            model_components: Maps between variable names and their component in the finite element space.

        Returns:
            The input dict now, where appropriate, holding gridfunctions in place of strings holding the paths to those
            gridfunctions.
        """

        for bc_type in bc_dict.keys():
            for var in bc_dict[bc_type].keys():
                if var in model_components:
                    component = model_components[var]
                else:
                    # For things like stress... All of these flux-type boundary conditions use the same finite element
                    # space as the velocity component.
                    component = model_components['u']
                for marker, val_lst in bc_dict[bc_type][var].items():
                    for val in val_lst:
                        if isinstance(val, str):
                            # Need to load a gridfunction from file.

                            # Check that the file exists.
                            val = self._find_rel_path_for_file(val)

                            # Use a component of the finite element space.
                            gfu_val = ngs.GridFunction(fes.components[component])

                            gfu_val.Load(val)

                            # Replace the value in the BC dictionary.
                            bc_dict[bc_type][var][marker] = gfu_val

                        else:
                            # Already parsed
                            pass

        return bc_dict

    def set_boundary_conditions(self, BC: Union[List[str], Dict[str, str]])\
            -> Tuple[Dict[str, Dict[str, Dict[str, ngs.CoefficientFunction]]], Dict[str, str]]:
        """
        Function to load the BC dictionary and to generate the list of Dirichlet BC markers for use when constructing
        the finite element space.

        Args:
            BC: The model's BC list (contains information about which types of BC are allowed).

        Returns:
            Tuple[Dict, Dict]:
                - BC_full: The full BC dictionary.
                - dirichlet_names: String containing all markers for Dirichlet BCs in NGSolve's required format.
        """

        # Need to modify a different dict than BC or there are conflicts between the conformal and DIM boundary
        # condition dictionaries.
        BC_full: Dict[str, Dict[str, Dict[str, ngs.CoefficientFunction]]] = {key: {} for key in BC}

        # Only use pre-defined BC types.
        for bc_type in BC:
            if bc_type in self.bc_dict.keys():
                BC_full[bc_type] = self.bc_dict[bc_type]

        # Additional work for storing information about strongly imposed Dirichlet conditions.
        dirichlet_names = {}
        if len(BC_full['dirichlet']) > 0:
            for var in BC_full['dirichlet']:
                if var == 'p':
                    # Pressure BCs are weakly imposed, thus we skip them.
                    pass
                else:
                    # Format the list of mesh markers for the Dirichlet BCs.
                    dirichlet_names[var] = '|'.join(list(BC_full['dirichlet'][var].keys()))

        if len(BC_full.get('pinned', {})) > 0:
            for var in BC_full['pinned']:
                if var in dirichlet_names.keys():
                    # Check that the variable doesn't already have Dirichlet BCs specified.
                    raise ValueError('Dirichlet boundary conditions have been specified for {0}. It is unnecessary to '
                                     'also pin the value of {0} at a point.'.format(var))

                # Format the list of mesh markers for the pinned BCs. These are later treated as Dirichlet BCs.
                dirichlet_names[var] = '|'.join(list(BC_full['pinned'][var].keys()))

        return BC_full, dirichlet_names

    def set_dirichlet_boundary_conditions(self,
                                          bc_dict: Dict[str, Dict[str, Dict[str, Union[List[CoefficientFunction],
                                                                                       List[list], List[float],
                                                                                       List[GridFunction]]]]],
                                          mesh: Mesh, gfu: GridFunction, model_components: Dict) \
            -> Dict[str, List[CoefficientFunction]]:
        """
        Function to load the strongly imposed Dirichlet BCs in order to apply them to the solution gridfunction.

        Args:
            bc_dict: Dictionary specifying the BCs and their mesh markers for each variable. The BCs are all given as
                floats, gridfunctions, or coefficientfunctions.
            mesh: The model's mesh.
            gfu: A gridfunction constructed on the model's finite element space.
            model_components: The model's model_components dictionary.

        Returns:
            Dictionary of coefficientfunctions used to set the strongly imposed Dirichlet BCs. Each dictionary entry
            holds a list, which gives the coefficientfunction for each time step in the time discretization scheme.
        """

        # Initialize empty
        g_D = {}

        if len(bc_dict['dirichlet']) > 0:
            for var in bc_dict['dirichlet']:

                # List of values for Dirichlet BCs for each variable
                dirichlet_lst: List = [[] for _ in self.t_param]

                if var == 'p':
                    # Pressure BCs are weakly imposed, thus we skip them.
                    pass
                else:
                    for marker in mesh.GetBoundaries():
                        # Get the Dirichlet BC value for this marker, or default to 0.0 if there isn't one. The 0.0
                        # will later be overwritten when the other types of boundary conditions are imposed.
                        component = model_components[var]

                        # TODO: This is a hack. There should be a better way of doing this, but ngs.FESpace doesn't seem
                        #  to have a dim argument that corresponds to the dimension of the space.
                        alternate: Union[List[float], List[Tuple]] # Just for type-hinting.

                        if gfu.components[component].dim == 1:
                            alternate = [0.0 for _ in self.t_param]
                        else:
                            alternate = [tuple([0.0] * gfu.components[component].dim) for _ in self.t_param]

                        val_lst = bc_dict['dirichlet'][var].get(marker, alternate)

                        # Store the value in a list.
                        for i in range(len(self.t_param)):
                            dirichlet_lst[i].append(val_lst[i])

                    # Format the list of mesh markers for the Dirichlet BCs.
                    g_D[var] = [ngs.CoefficientFunction(tmp) for tmp in dirichlet_lst]

        if len(bc_dict.get('pinned', {})) > 0:
            for var in bc_dict['pinned']:
                if var in g_D.keys():
                    # Check that the variable doesn't already have Dirichlet BCs specified.
                    raise ValueError('Dirichlet boundary conditions have been specified for {0}. It is unnecessary to '
                                     'also pin the value of {0} at a point.'.format(var))

                # List of values for pinned BCs for each variable
                pinned_lst: List = [[] for _ in self.t_param]

                for marker in mesh.GetBoundaries():
                    # Get the pinned BC value for this marker, or default to 0.0 if there isn't one. The 0.0
                    # will later be overwritten when the other types of boundary conditions are imposed.
                    component = model_components[var]

                    # TODO: This is a hack. There should be a better way of doing this, but ngs.FESpace doesn't seem
                    #  to have a dim argument that corresponds to the dimension of the space.
                    if gfu.components[component].dim == 1:
                        alternate = [0.0 for _ in self.t_param]
                    else:
                        alternate = [tuple([0.0] * gfu.components[component].dim) for _ in self.t_param]

                    val_lst = bc_dict['pinned'][var].get(marker, alternate)

                    # Store the value in a list.
                    for i in range(len(self.t_param)):
                        pinned_lst[i].append(val_lst[i])

                # Format the list of mesh markers for the pinned BCs.
                g_D[var] = [ngs.CoefficientFunction(tmp) for tmp in pinned_lst]

        return g_D

    def update_boundary_conditions(self, t_param: List[Parameter],
                                   updated_variables: List[Dict[str, Union[float, CoefficientFunction, GridFunction]]],
                                   mesh: Mesh) -> None:
        """
        Function to update the boundary conditions with new values of the model_variables.

        Args:
            t_param: List of parameters representing the current time and previous time steps.
            updated_variables: List of dictionaries containing any new model variables and their values at each time
                step used in the time discretization scheme.
            mesh: Mesh used by the model
        """

        for k1, v1 in self.bc_re_parse_dict.items():
            for k2, v2 in self.bc_re_parse_dict[k1].items():
                self.bc_dict[k1][k2] = self.re_parse(self.bc_dict[k1][k2], self.bc_re_parse_dict[k1][k2],
                                                     t_param,updated_variables, mesh)
