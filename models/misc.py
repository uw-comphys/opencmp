"""
Module containing helper functions related to models.
"""
import models
from models import Model

from sys import modules
from typing import Type


def get_model_class(model_name: str) -> Type[Model]:
    """
    Function to find the correct model to
    to find, initialize, and return an instance of the desired model class(es).

    Will raise a ValueError if a model could not be found.

    Args:
        model_name: String representing the model to use, must be the name of the model class

    Returns:
        ~: An initialized instance of the desired model.
    """

    # Find the class by it's name
    model_class = getattr(modules[models.__name__], model_name)

    # Check that whatever we found is actually a model
    if not issubclass(model_class, Model):
        # Raise an error if it isn't
        raise AttributeError("Provided class is not a model")

    return model_class
