"""
Copyright 2021 the authors (see AUTHORS file for full list)

This file is part of OpenCMP.

OpenCMP is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

OpenCMP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with OpenCMP.  If not, see <https://www.gnu.org/licenses/>.
"""

import run
import math
import multiprocessing
from typing import Callable, Tuple
from pytest import CaptureFixture


def run_example(configfile_path: str) -> None:
    """
    Function to run an example given the path to the config file.

    Args:
        configfile_path: File path relative to OpenCMP/ for the config file
    """
    return run.main(configfile_path)


def manual_output_check(capsys: CaptureFixture, test_name: str, configfile_path: str) -> None:
    """
    Use for examples that have extensive error analysis (ex: the table produced by convergence testing).
    The error analysis output should be checked manually for correctness
    since there doesn't seem to be an easy way to parse it.

    Args:
        capsys: Pytest object used to get access to stdout and stderr
        test_name: The name of the current test being run, used for writing to console
        configfile_path: File path relative to OpenCMP/ for the config file
    """
    run_example(configfile_path)
    captured = capsys.readouterr()
    with capsys.disabled():
        print(test_name + ' output')
        print(captured.out)


def automated_output_check(capsys: CaptureFixture, configfile_path: str, expected_err: float) -> None:
    """
    Use for examples that have a single line of error analysis (ex: one l2 norm).
    The error analysis output is automatically checked for correctness.

    Args:
        capsys: Pytest object used to get access to stdout and stderr
        configfile_path: File path relative to OpenCMP/ for the config file
        expected_err: The value of the expected error for that run
    """
    run_example(configfile_path)
    captured = capsys.readouterr()

    output = captured.out
    err_loc = output[::-1].find(' ')
    err = float(output[-err_loc:])

    assert math.isclose(err, expected_err, rel_tol=1)


def timed_output_check(func: Callable, args: Tuple, timeout: float) -> None:
    """
    Use for very long transient examples to test that they initialize properly without running the full example.

    Args:
        func: The function being called to initialize the run.
        args: The function's arguments.
        timeout: How long to run the test before timing out.
    """
    p = multiprocessing.Process(target=func, args=args)
    p.start()

    p.join(timeout)

    if p.is_alive():
        # The test is still running so exit it and return that it passed.
        p.terminate()
        p.join()
    else:
        # Check if the test exited on an error or just finished faster than expected.
        assert p.exitcode == 0


class TestStationaryExamples:
    """ Class to test the stationary examples. """
    def test_poisson_1(self, capsys: CaptureFixture) -> None:
        manual_output_check(capsys, 'poisson_1', 'Example Runs/Poisson/poisson_1/config')

    def test_poisson_2(self, capsys: CaptureFixture) -> None:
        manual_output_check(capsys, 'poisson_2', 'Example Runs/Poisson/poisson_2/config')

    def test_stokes_1(self, capsys: CaptureFixture) -> None:
        manual_output_check(capsys, 'stokes_1', 'Example Runs/Stokes/stokes_1/config')

    def test_stokes_2(self, capsys: CaptureFixture) -> None:
        manual_output_check(capsys, 'stokes_2', 'Example Runs/Stokes/stokes_2/config')

    def test_ins_3(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/INS/ins_3/config', 2e-14)

    def test_ins_4(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/INS/ins_4/config', 4e-10)

    def test_ins_5(self, capsys: CaptureFixture) -> None:
        manual_output_check(capsys, 'ins_5', 'Example Runs/INS/ins_5/config')

    def test_ins_6(self, capsys: CaptureFixture) -> None:
        manual_output_check(capsys, 'ins_6', 'Example Runs/INS/ins_6/config')

    def test_DIM_poisson_1(self) -> None:
        run_example('Example Runs/DIM/Poisson/DIM_poisson_1/config')

    def test_DIM_poisson_2(self) -> None:
        run_example('Example Runs/DIM/Poisson/DIM_poisson_2/config')

    def test_DIM_poisson_3(self) -> None:
        run_example('Example Runs/DIM/Poisson/DIM_poisson_3/config')

    def test_DIM_poisson_4(self) -> None:
        run_example('Example Runs/DIM/Poisson/DIM_poisson_4/config')

    def test_DIM_poisson_5(self) -> None:
        run_example('Example Runs/DIM/Poisson/DIM_poisson_5/config')

    def test_DIM_poisson_6(self) -> None:
        run_example('Example Runs/DIM/Poisson/DIM_poisson_6/config')

    def test_DIM_stokes_1(self) -> None:
        run_example('Example Runs/DIM/Stokes/DIM_stokes_1/config')

    def test_DIM_stokes_2(self) -> None:
        run_example('Example Runs/DIM/Stokes/DIM_stokes_2/config')


class TestTransientExamples:
    """ Class to test the transient examples. """

    def test_poisson_3(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_3/config', 9e-6)

    def test_poisson_4(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_4/config', 9e-6)

    def test_poisson_5(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_5/config', 9e-6)

    def test_poisson_6(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_6/config', 9e-6)

    def test_poisson_7(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_7/config', 9e-6)

    def test_poisson_8(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_8/config', 9e-6)

    def test_poisson_9(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_9/config', 1e-5)

    def test_poisson_10(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_10/config', 1e-5)

    def test_poisson_11(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_11/config', 1e-5)

    def test_poisson_12(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_12/config', 1e-5)

    def test_ins_7(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/INS/ins_7/config', 1e-4)

    def test_ins_8(self, capsys: CaptureFixture) -> None:
        automated_output_check(capsys, 'Example Runs/INS/ins_7/config', 1e-4)


class TestSlowTransientExamples:
    """
    Class to test the slow transient examples. All of these examples are forced to end after 10s, so these tests only
    confirm that the examples initialize properly, it will not catch errors that occur after several time steps.
    """

    def test_ins_1_IC(self) -> None:
        run_example('Example Runs/INS/ins_1/config_IC')

    def test_ins_1(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_1/config',), 10)

    def test_ins_2_IC(self) -> None:
        run_example('Example Runs/INS/ins_2/config_IC')

    def test_ins_2(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_2/config',), 10)

    def test_ins_9_IC(self) -> None:
        run_example('Example Runs/INS/ins_9/config_IC')

    def test_ins_9(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_9/config',), 10)

    def test_ins_10_IC(self) -> None:
        run_example('Example Runs/INS/ins_10/config_IC')

    def test_ins_10(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_10/config',), 10)

    def test_ins_11_IC(self) -> None:
        run_example('Example Runs/INS/ins_11/config_IC')

    def test_ins_11(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_11/config',), 10)

    def test_ins_12_IC(self) -> None:
        run_example('Example Runs/INS/ins_12/config_IC')

    def test_ins_12(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_12/config',), 10)

    def test_ins_13_IC(self) -> None:
        run_example('Example Runs/INS/ins_13/config_IC')

    def test_ins_13(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_13/config',), 10)

    def test_ins_14_IC(self) -> None:
        run_example('Example Runs/INS/ins_14/config_IC')

    def test_ins_14(self) -> None:
        timed_output_check(run_example, ('Example Runs/INS/ins_14/config',), 10)