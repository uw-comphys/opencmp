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

from numpy import isclose
from multiprocessing import Process
from pytest import CaptureFixture
from opencmp.config_functions import ConfigParser
from opencmp.run import run
from typing import Callable, List, Tuple


def run_example(config: ConfigParser) -> None:
    """
    Function to run an example given the path to the config file.

    Args:
        config: An initialized config parser holding the relevant information for the simulation
    """
    return run("", config)


# TODO: rename to something better
def manual_output_check(capsys: CaptureFixture, test_name: str, config: ConfigParser) -> None:
    """
    Use for examples that have extensive error analysis (ex: the table produced by convergence testing).

    The error analysis output should be checked manually for correctness since there doesn't seem to be an easy way to
    parse it.

    Args:
        capsys: Pytest object used to get access to stdout and stderr.
        test_name: The name of the current test being run, used for writing to console.
        config: An initialized config parser holding the relevant information for the simulation
    """
    run_example(config)
    captured = capsys.readouterr()
    with capsys.disabled():
        print(test_name + ' output')
        print(captured.out)


# TODO: rename to something better
def automated_output_check(capsys: CaptureFixture, config: ConfigParser, expected_err: List[float]) -> None:
    """
    Use for examples that have a simple lines of error analysis (NOT convergence tests).

    The error analysis output is automatically checked for correctness.

    Args:
        capsys: Pytest object used to get access to stdout and stderr.
        config: An initialized config parser holding the relevant information for the simulation
        expected_err: A list of floats representing the expected error values
    """
    run_example(config)
    captured = capsys.readouterr()

    # Pass through solver output
    print("")
    [print(line) for line in captured.out.split('\n')[-(len(expected_err)+1):]]

    # Grab error values from console output and convert into float
    errors = [float(line.split(': ')[1]) for line in captured.out.split('\n')[-(len(expected_err)+1):-1]]

    # For each error, check it against the provided value
    for i in range(len(expected_err)):
        if not isclose(errors[i], expected_err[i], rtol=3, atol=0):

            print('{}th Expected error: {}'.format(i+1, expected_err[i]))
            print('{}th Actual   error: {}'.format(i+1, errors[i]))
            assert False


def timed_output_check(func: Callable, args: Tuple, timeout: float) -> None:
    """
    Use for very long transient examples to test that they initialize properly without running the full example.

    Args:
        func: The function being called to initialize the run.
        args: The function's arguments.
        timeout: How long to run the test before timing out.
    """
    p = Process(target=func, args=args)
    p.start()

    p.join(timeout)

    if p.is_alive():
        # The test is still running so exit it and return that it passed.
        p.terminate()
        p.join()
    else:
        # Check if the test exited on an error or just finished faster than expected.
        assert p.exitcode == 0
