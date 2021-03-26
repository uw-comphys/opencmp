import pytest
import run
import math


def run_example(configfile_path):
    """ Function to run an example given the path to the config file. """
    return run.main(configfile_path)


def manual_output_check(capsys, test_name, configfile_path):
    """
    Use for examples that have extensive error analysis (ex: the table produced by convergence testing). The error
    analysis output should be checked manually for correctness since there doesn't seem to be an easy way to parse it.
    """
    run_example(configfile_path)
    captured = capsys.readouterr()
    with capsys.disabled():
        print(test_name + ' output')
        print(captured.out)


def automated_output_check(capsys, configfile_path, expected_err):
    """
    Use for examples that have a single line of error analysis (ex: one l2 norm). The error analysis output is
    automatically checked for correctness.
    """
    run_example(configfile_path)
    captured = capsys.readouterr()

    output = captured.out
    err_loc = output[::-1].find(' ')
    err = float(output[-err_loc:])

    assert math.isclose(err, expected_err, rel_tol=1)


class TestStationaryExamples:
    """ Class to test the stationary examples. """
    def test_poisson_1(self, capsys):
        manual_output_check(capsys, 'poisson_1', 'Example Runs/Poisson/poisson_1/config')

    def test_poisson_2(self, capsys):
        manual_output_check(capsys, 'poisson_2', 'Example Runs/Poisson/poisson_2/config')

    def test_stokes_1(self, capsys):
        manual_output_check(capsys, 'stokes_1', 'Example Runs/Stokes/stokes_1/config')

    def test_stokes_2(self, capsys):
        manual_output_check(capsys, 'stokes_1', 'Example Runs/Stokes/stokes_2/config')

    def test_ins_3(self, capsys):
        automated_output_check(capsys, 'Example Runs/INS/ins_3/config', 2e-14)

    def test_ins_4(self, capsys):
        automated_output_check(capsys, 'Example Runs/INS/ins_4/config', 4e-10)

    def test_ins_5(self, capsys):
        manual_output_check(capsys, 'ins_5', 'Example Runs/INS/ins_5/config')

    def test_ins_6(self, capsys):
        manual_output_check(capsys, 'ins_6', 'Example Runs/INS/ins_6/config')

    def test_DIM_poisson_1(self):
        run_example('Example Runs/DIM/Poisson/DIM_poisson_1/config')

    def test_DIM_poisson_2(self):
        run_example('Example Runs/DIM/Poisson/DIM_poisson_2/config')

    def test_DIM_poisson_3(self):
        run_example('Example Runs/DIM/Poisson/DIM_poisson_3/config')

    def test_DIM_poisson_4(self):
        run_example('Example Runs/DIM/Poisson/DIM_poisson_4/config')

    def test_DIM_poisson_5(self):
        run_example('Example Runs/DIM/Poisson/DIM_poisson_5/config')

    def test_DIM_poisson_6(self):
        run_example('Example Runs/DIM/Poisson/DIM_poisson_6/config')


@pytest.mark.slow
class TestTransientExamples:
    """ Class to test the transient examples. These tend to be slow. """
    def test_poisson_3(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_3/config', 9e-6)

    def test_poisson_4(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_4/config', 9e-6)

    def test_poisson_5(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_5/config', 9e-6)

    def test_poisson_6(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_6/config', 9e-6)

    def test_poisson_7(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_7/config', 9e-6)

    def test_poisson_8(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_8/config', 9e-6)

    def test_poisson_9(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_9/config', 1e-5)

    def test_poisson_10(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_10/config', 1e-5)

    def test_poisson_11(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_11/config', 1e-5)

    def test_poisson_12(self, capsys):
        automated_output_check(capsys, 'Example Runs/Poisson/poisson_12/config', 1e-5)

    def test_DIM_stokes_1(self):
        run_example('Example Runs/DIM/Stokes/DIM_stokes_1/config')

    def test_DIM_stokes_2(self):
        run_example('Example Runs/DIM/Stokes/DIM_stokes_2/config')
