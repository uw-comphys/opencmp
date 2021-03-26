from typing import Union
import ngsolve as ngs
from models import Model
from pathlib import Path


class SolutionFileSaver:
    """
    Class to handle the saving of GridFunctions and CoefficientFunctions to file
    """
    def __init__(self, model: Model, quiet: bool = False) -> None:
        """
        Initializer

        Args:
            model: The model being solved from which to get necessary information.
            quiet: If True suppresses the warning about the default value being used for a parameter.
        """

        # Check that only valid output types were passed
        base_type = model.config.get_item(['VISUALIZATION', 'save_type'], str, quiet)
        if base_type not in ['.sol', '.vtu']:
            print('Can\'t output to file type {}.'.format(base_type))

        self.save_dir = model.config.get_item(['OTHER', 'run_dir'], str, quiet) + '/output/'
        self.base_filename = self.save_dir + model.name + "_"
        self.base_subdivision = model.config.get_item(['VISUALIZATION', 'subdivision'], int, quiet)

        # Create the save dir if it doesn't exist
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # NOTE: -1 is the value used whenever an int default is needed.
        if self.base_subdivision == -1:
            self.base_subdivision = model.interp_ord

    def save(self, gfu: Union[ngs.GridFunction, ngs.CoefficientFunction], timestep: float) -> None:
        """
        Function to save the provided GridFunction or CoefficientFunction to file.

        Args:
            gfu: GridFunction or CoefficientFunction to save
            timestep: the current time step, used for naming the file
        """

        # Assemble filename
        filename = self.base_filename + str(timestep) + '.sol'

        # Save to file
        gfu.Save(filename)
