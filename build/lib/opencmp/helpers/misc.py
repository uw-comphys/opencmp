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

from typing import Dict
import importlib.util
import sys


def merge_bc_dict(dict_add_to: Dict, dict_add_from: Dict) -> Dict:
    """
    This function merges two boundary condition dictionaries recursively.

    The value of each dictionary is either another dictionary or a list.

    The merging will be done semi shallowly, so not

    If dict_add_from contains a key which is not present in dict_add_to, dict_add_from[key] is copied, shallowly.
    If the key is present in both dictionary, then the type of value associated with it must be the same.
    I.e. type(dict_add_to[i]) == type(dict_add_from[i])

    If it is a dictionary, the function is called recursively to merge the dictionaries and store them in dict_add_to.

    If the entries are lists, then the entries will merged according to the following rules.
    Let: list_add_to come from dict_add_to and list_add_from come from dict_add_from.
    1. list_add_from[i] is None: leave list_add_to[i] alone
    2. Otherwise, override list_add_to[i] with list_add_from[i]

    The two lists must be of the same length

    Args:
        dict_add_to: The main dictionary into which things are added
        dict_add_from: The dictionary from which things are added

    Return:
        The dict_add_to dictionary with the new pieces added to it.
    """

    for key in dict_add_from:
        if key in dict_add_to:
            if not isinstance(dict_add_from[key], type(dict_add_to[key])):
                raise ValueError("Dictionaries are supposed to have the same types for the same key.")

            if type(dict_add_to[key]) is dict:
                dict_add_to[key] = merge_bc_dict(dict_add_to[key], dict_add_from[key])
            else:
                if len(dict_add_to[key]) != len(dict_add_from[key]):
                    raise ValueError("Lists are not of the same length.")

                tmp_lst = []
                for i in range(len(dict_add_to[key])):
                    tmp_lst.append(dict_add_from[key][i] if dict_add_from[key][i] is not None else dict_add_to[key][i])
                dict_add_to[key] = tmp_lst
        else:
            dict_add_to[key] = dict_add_from[key]

    return dict_add_to


def can_import_module(module_name: str) -> bool:
    """
    This function checks if a module can be imported.
    From: https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported

    Args:
        module_name: The name of the module to be imported

    Return:
        True if the module can be imported, False otherwise
    """

    if module_name in sys.modules:
        return True
    elif importlib.util.find_spec(module_name) is not None:
        return True
    else:
        return False
