from config_functions import ConfigParser
from ngsolve import GridFunction, Mesh, Parameter, VTKOutput
from models import get_model_class, Model
from pathlib import Path
from os import remove
import meshio
from typing import Union
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


def sol_to_vtu(output_dir_path: str, config_file_path: str, model: Union[Model, None] = None,
               delete_sol_file: bool = False) -> None:
    """
    Function to take the output .sol files and convert them into .vtu for visualization.

    Args:
        output_dir_path: The path to the folder in which the .sol files are, and where the .vtu files will be saved.
        config_file_path: The path to the config file used by the model.
        model: The model that generated the .sol files.
        delete_sol_file: Bool to indicate whether or not to delete the original .sol files after converting to .vtu,
                         Default is False.
    """
    # Create config parser
    config = ConfigParser(config_file_path)

    # Being run outside of run.py, so have to create model
    if model is None:
        # Load model
        model_name  = config.get_item(['OTHER', 'model'], str)
        model_class = get_model_class(model_name)
        model       = model_class(config, Parameter(0.0))

    # Number of subdivisions per element
    subdivision = config.get_item(['VISUALIZATION', 'subdivision'], int)
    # NOTE: -1 is the value used whenever an int default is needed.
    if subdivision == -1:
        subdivision = model.interp_ord

    # Generate a list of all .sol files
    sol_path_generator  = Path(output_dir_path).rglob('*.sol')
    sol_path_list       = [str(sol_path) for sol_path in sol_path_generator]

    # Number of files to convert
    n_files = len(sol_path_list)
    # Number of cores to use
    # NOTE: No point of starting more threads than files, and also lets us depend on modulo math later.
    n_threads = min(n_files, cpu_count())

    # Create gridfunctions, one per thread
    gfus = [model.construct_gfu() for _ in range(n_threads)]

    # Create a list to contain the .pvd entries
    output_list = ['' for _ in range(n_files)]

    # NOTE: We HAVE to use Pool, and not ThreadPool. ThreadPool causes seg faults on the VTKOutput call.
    with Pool(processes=n_threads) as pool:
        # Create the pool and start it. It will automatically take and run the next entry when it needs it
        a = [
                pool.apply_async(_sol_to_vtu, (gfus[i % n_threads], sol_path_list[i], output_dir_path,
                                               model.save_names, delete_sol_file, subdivision, model.mesh)
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
    with open(output_dir_path + 'transient.pvd', 'a+') as file:
        for line in output_list:
            file.write(line)


def _sol_to_vtu(gfu: GridFunction, sol_path_str: str, output_dir_path: str,
                save_names: str, delete_sol_file: bool, subdivision: int, mesh: Mesh) -> str:
    """
    Function that gets parallelized and does the actual sol-to-vtu conversion.

    Args:
        gfu: The grid function into which to load the .sol file
        sol_path_str: The path to the solve file to load
        output_dir_path: The path to the directory to save the .vtu into
        save_names: The names of the variables to save
        delete_sol_file: Whether or not to detele the sol file after
        subdivision: Number of subdivisions on each mesh element
        mesh: The mesh on which the gfu was solved.

    Returns:
        ~: A string containing the entry for the .pvd file for this .vtu file.
    """

    # Get the timestep for this .sol file from its name
    sol_name = sol_path_str.split('/')[-1][:-4]
    time_str = sol_name.split('_')[-1]

    # Name for the .vtu
    filename = output_dir_path + sol_name

    # Load data into gfu
    gfu.Load(sol_path_str)

    # Convert gfu components into form needed for VTKOutput
    if len(gfu.components) > 0:
        coefs = [component for component in gfu.components]
    else:
        coefs = [gfu]

    # Write to .vtk
    VTKOutput(ma=mesh, coefs=coefs, names=save_names,
              filename=filename, subdivision=subdivision).Do()

    # Convert .vtk to .vtu
    meshio.read(filename + '.vtk').write(filename + '.vtu')

    # Remove the .vtk
    remove(filename + '.vtk')

    # Delete .sol
    if delete_sol_file:
        remove(sol_path_str)

    # Write timestep in .pvd
    return '<DataSet timestep=\"%e\" group=\"\" part=\"0\" file=\"%s\"/>\n'\
           % (float(time_str), filename.split('/')[-1] + '.vtu')
