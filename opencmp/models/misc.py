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

from . import Model, models_dict

from typing import Type

"""
Module containing helper functions related to models.
"""


def get_model_class(model_name: str) -> Type[Model]:
    """
    Function to find the correct model to to find, initialize, and return an instance of the desired model class(es).

    Will raise a ValueError if a model could not be found.

    Args:
        model_name: String representing the model to use, must be the name of the model class.

    Returns:
        An initialized instance of the desired model.
    """

    # Find the class by it's name
    model_class = models_dict[model_name]

    # Check that whatever we found is actually a model
    if not issubclass(model_class, Model):
        # Raise an error if it isn't
        raise AttributeError("Provided class is not a model")

    return model_class
