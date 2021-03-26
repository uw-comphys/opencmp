import sys
from models import get_model_class
from solvers import get_solver_class
from error_analysis import h_convergence, p_convergence
from helpers.error import calc_error
from config_functions import ConfigParser
import pyngcore as ngcore
from ngsolve import ngsglobals
from helpers.post_processing import sol_to_vtu


def main(config_file_path: str) -> None:
    """
    Main function that runs OpenCMP.

    Args:
        config_file_path: Filename of the config file to load
    """
    # Load the config file.
    config = ConfigParser(config_file_path)

    # Load run parameters from the config file.
    num_threads = config.get_item(['OTHER', 'num_threads'], int)
    msg_level = config.get_item(['OTHER', 'messaging_level'], int, quiet=True)
    model_name = config.get_item(['OTHER', 'model'], str)

    # Load error analysis parameters from the config file.
    check_error = config.get_item(['ERROR ANALYSIS', 'check_error'], bool)
    # Suppressing the warning about using the default value for convergence_test.
    convergence_test = config.get_item(['ERROR ANALYSIS', 'convergence_test'], str, quiet=True)

    # Set parameters for ngsolve
    ngcore.SetNumThreads(num_threads)
    ngsglobals.msg_level = msg_level

    # Run the model.
    with ngcore.TaskManager():

        model_class = get_model_class(model_name)
        solver_class = get_solver_class(config)
        solver = solver_class(model_class, config)
        sol = solver.solve()

        if check_error:
            calc_error(config, solver.model, sol)

        if convergence_test == 'h':
            h_convergence(config, solver, sol)
        elif convergence_test == 'p':
            p_convergence(config, solver, sol)

    save_output = config.get_item(['VISUALIZATION', 'save_to_file'], str, quiet=True)
    if save_output:
        save_type = config.get_item(['VISUALIZATION', 'save_type'], str, quiet=True)

        # Run the post-processor to convert the .sol to .vtu
        if save_type == '.vtu':
            print('Converting saved output to VTU.')

            # Path where output is stored
            output_dir_path = config.get_item(['OTHER', 'run_dir'], str) + '/output/'

            # Run the conversion
            sol_to_vtu(output_dir_path, config_file_path, solver.model)

    return


if __name__ == '__main__':
    main(sys.argv[1])
