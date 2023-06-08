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

from ..config_functions import ConfigParser
from ngsolve import GridFunction, Mesh, Parameter, VTKOutput, H1
from ..models import get_model_class, Model
from pathlib import Path
from os import remove
from typing import Optional, Union
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

from ..solvers import Solver


class PhaseFieldModelMimic:
    """
    Helper class that mimics the attributes of the Model class that are used during saving and post-processing in order
    to be able to post-process saved phase field .sol files to .vtu.
    """

    def __init__(self, model_to_copy: Model) -> None:
        """
        Initializer

        Args:
            model_to_copy: The model for the general simulation. Needed in order to know the config parameter values.
        """

        self.interp_ord = model_to_copy.interp_ord
        self.save_names = ['phi']
        self.mesh = model_to_copy.mesh
        self.fes = H1(self.mesh, order=self.interp_ord)

    def construct_gfu(self) -> GridFunction:
        """ Function to construct a phase field gridfunction. """
        return GridFunction(self.fes)


def sol_to_components(config_parser: ConfigParser, output_dir_path: str, model: Optional[Model] = None) -> None:
    """
    Function which takes the final .sol file in the output_dir and splits each component out into its own .sol file.
    Allows for easily loading only part of a previous solution into a different simulation.
    Particularly useful for loading a steady-state velocity profile into the multi-component INS model.
    .sol files for the individual components will be named as C_t.sol where C is the component name from model.components
    and t is the time obtained from the filename of the original .sol file.

    Args:
        config_parser:      The loaded config parser used by the model
        output_dir_path:    The path to the folder in which the .sol files are, and where the .vtu files will be saved.
        model:              The model that generated the .sol files.
                            If one is not provided (such as when running manually), one is created from the config_parser.
    """

    # Being run outside of run_post_processing, so have to create model
    if model is None:
        # Load model
        model_name = config_parser.get_item(['OTHER', 'model'], str)
        model_class = get_model_class(model_name)
        model = model_class(config_parser, [Parameter(0.0)])

    # Ensure that the output folder exists
    sol_component_path = output_dir_path + 'components_sol/'
    Path(sol_component_path).mkdir(parents=True, exist_ok=True)

    # Generate a list of all .sol file paths
    sol_paths_all = [str(sol_path) for sol_path in Path(output_dir_path + model.name + '_sol/').rglob('*.sol')]
    sol_path_final = ""
    split_str = model.name + "_"
    time_max = -1.0
    for sol_path in filter(lambda path: model.name in path, sol_paths_all):
        time_i = float(sol_path.split(split_str)[2][:-4])
        if time_i > time_max:
            time_max = time_i
            sol_path_final = sol_path

    if len(sol_path_final) == 0:
        raise FileNotFoundError('A .sol file for model \"' + model.name
                                + '\" was not found in folder \"' + output_dir_path + 'sol/' + '\"')

    # Get the timestep for this .sol file from its name
    sol_name = sol_path_final.split('/')[-1][:-4]

    gfu_for_saving = model.construct_gfu()
    gfu_for_saving.Load(sol_path_final)

    for component_name in model.model_components:
        gfu_for_saving.components[model.model_components[component_name]].Save(sol_component_path + component_name + '.sol')


def sol_to_vtu(config_parser: ConfigParser, solver: Solver) -> None:
    """
    Wrapper function to handle the VTU-to-SOL conversion.

    Args:
        config_parser:  The config_parser for the simulation files to convert to .vtu
        solver:         The solver object created from the config_parser which produced the .sol files
    """
    # Path where output is stored
    output_dir_path = config_parser.get_item(['OTHER', 'run_dir'], str) + '/output/'

    # Run the conversion
    sol_to_vtu_direct(config_parser, output_dir_path, solver.model)

    # Repeat for the saved phase field .sol files if using the diffuse interface method.
    if solver.model.DIM:
        print('Converting saved phase fields to VTU.')

        # Construct a mimic of the Model class appropriate for the phase field (mainly contains the correct
        # finite element space).
        phi_model = PhaseFieldModelMimic(solver.model)

        # Path where the output is stored
        output_dir_phi_path = config_parser.get_item(['OTHER', 'run_dir'], str) + '/output_phi/'

        # Run the conversion.
        # Note: The normal main simulation ConfigParse can be used since it is only used
        # to get a value for subdivision.
        sol_to_vtu_direct(config_parser, output_dir_phi_path, phi_model)


def sol_to_vtu_direct(config_parser: ConfigParser, output_dir_path: str, model: Optional[Union[Model, PhaseFieldModelMimic]] = None,
                      delete_sol_file: bool = False, allow_all_threads: bool = False) -> None:
    """
    Function to take the output .sol files and convert them into .vtu for visualization.

    Args:
        config_parser:      The loaded config parser used by the model
        output_dir_path:    The path to the folder in which the .sol files are, and where the .vtu files will be saved.
        model:              The model that generated the .sol files.
                            If one is not provided (such as when running manually), one is created from the config_parser.
        delete_sol_file:    Bool to indicate if to delete the original .sol files after converting to .vtu,
                            Default is False.
        allow_all_threads:  Bool to indicate if to use all available threads (if there are enough files),
                            Default is False.
    """

    # Being run outside of run_post_processing, so have to create model
    if model is None:
        # Load model
        model_name  = config_parser.get_item(['OTHER', 'model'], str)
        dim_used    = config_parser.get_item(['DIM', 'diffuse_interface_method'], bool)
        model_class = get_model_class(model_name, dim_used)
        model       = model_class(config_parser, [Parameter(0.0)])

    # Number of subdivisions per element
    subdivision = config_parser.get_item(['VISUALIZATION', 'subdivision'], int)
    # NOTE: -1 is the value used whenever an int default is needed.
    if subdivision == -1:
        subdivision = model.interp_ord

    # Generate a list of all .sol files
    sol_path_generator  = Path(output_dir_path + model.name + '_sol/').rglob('*' + model.name + '*.sol')
    sol_path_list       = [str(sol_path) for sol_path in sol_path_generator]

    # Number of files to convert
    n_files = len(sol_path_list)

    # Figure out the maximum number of threads at our disposal
    if allow_all_threads:
        # Number of threads on the machine that we have access to
        num_threads = cpu_count()
    else:
        # Number of threads to use as specified in the config file
        num_threads = config_parser.get_item(['OTHER', 'num_threads'], int)

    # Number of threads to use
    # NOTE: No point of starting more threads than files, and also lets us depend on modulo math later.
    n_threads = min(n_files, num_threads)

    # Create gridfunctions, one per thread
    gfus = [model.construct_gfu() for _ in range(n_threads)]

    # Create a list to contain the .pvd entries
    output_list = ['' for _ in range(n_files)]

    # NOTE: We HAVE to use Pool, and not ThreadPool. ThreadPool causes seg faults on the VTKOutput call.
    with Pool(processes=n_threads) as pool:
        # Create the pool and start it. It will automatically take and run the next entry when it needs it
        a = [
                pool.apply_async(_sol_to_vtu_parallel_runner, (gfus[i % n_threads], sol_path_list[i], output_dir_path,
                                                               model.save_names, model.name, delete_sol_file,
                                                               subdivision, model.mesh)
                                 ) for i in range(n_files)
        ]

        # Iterate through each thread and get it's result when it's done
        for i in range(len(a)):
            # Grab the result string and insert it in the correct place in the output list
            output_list[i] = a[i].get()

    # Add the header and footer
    output_list.insert(0, '<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\"\n' +
                       'byte_order=\"LittleEndian\"\ncompressor=\"vtkZLibDataCompressor\">\n<Collection>\n')
    output_list.append('</Collection>\n</VTKFile>')

    # Write each line to the file
    with open(output_dir_path + model.name + '_transient.pvd', 'a+') as file:
        for line in output_list:
            file.write(line)


def _sol_to_vtu_parallel_runner(gfu: GridFunction, sol_path_str: str, output_dir_path: str, save_names: str,
                                model_name: str, delete_sol_file: bool, subdivision: int, mesh: Mesh) -> str:
    """
    Function that gets parallelized and does the actual sol-to-vtu conversion.

    Args:
        gfu:                The grid function into which to load the .sol file
        sol_path_str:       The path to the solve file to load
        output_dir_path:    The path to the directory to save the .vtu into
        save_names:         The names of the variables to save
        model_name:         The name of the model
        delete_sol_file:    Whether or not to delete the sol file after
        subdivision:        Number of subdivisions on each mesh element
        mesh:               The mesh on which the gfu was solved.

    Returns:
        A string containing the entry for the .pvd file for this .vtu file.
    """

    # Get the timestep for this .sol file from its name
    sol_name = sol_path_str.split('/')[-1][:-4]
    time_str = sol_name.split('_')[-1]

    # Name for the .vtu
    filename = output_dir_path + model_name + '_vtu/' + sol_name

    # Load data into gfu
    gfu.Load(sol_path_str)

    # Convert gfu components into form needed for VTKOutput
    if len(gfu.components) > 0:
        coefs = [component for component in gfu.components]
    else:
        coefs = [gfu]

    # Write to .vtu
    VTKOutput(ma=mesh, coefs=coefs, names=save_names,
              filename=filename, subdivision=subdivision).Do()

    # Check that the file was created
    if not Path(filename + '.vtu').exists():
        raise FileNotFoundError('Neither .vtk nor .vtu files are being generated. Something is wrong with _sol_to_vtu_parallel_runner.')

    # Delete .sol
    if delete_sol_file:
        remove(sol_path_str)

    # Write timestep in .pvd
    return '<DataSet timestep=\"%e\" group=\"\" part=\"0\" file=\"%s\"/>\n'\
           % (float(time_str), model_name + '_vtu/' + filename.split('/')[-1] + '.vtu')
