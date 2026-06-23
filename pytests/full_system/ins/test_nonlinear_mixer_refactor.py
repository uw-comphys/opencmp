"""
Characterization tests for the nonlinear mixer extraction refactor.

All three schemes (Anderson/default, LinearMixing, DiagBroyden) must converge to
the same solution on the pipe-flow case.  Expected errors are the same as those
in test_ins.py for the CG case — the nonlinear solver scheme affects the path to
convergence, not the converged solution itself.

If any test here fails after a code change, the mixing math diverged from the
pre-refactor behaviour.
"""

import pytest
from pytest import CaptureFixture, fixture
from opencmp.helpers.testing import automated_output_check
from opencmp.config_functions import ConfigParser

EXPECTED_ERRORS = [2e-7, 5e-8, 1.5e-4, 3.5e-5]


@fixture
def pipe_velocity_flow() -> ConfigParser:
    return ConfigParser('pytests/full_system/ins/pressure_flow_in_pipe_velocity/config')


class TestNonlinearMixerRefactor:
    def test_anderson_default(self, capsys: CaptureFixture, pipe_velocity_flow: ConfigParser) -> None:
        automated_output_check(capsys, pipe_velocity_flow, EXPECTED_ERRORS)

    def test_linear_mixing(self, capsys: CaptureFixture, pipe_velocity_flow: ConfigParser) -> None:
        pipe_velocity_flow['SOLVER']['nonlinear_solver'] = 'LinearMixing'
        automated_output_check(capsys, pipe_velocity_flow, EXPECTED_ERRORS)

    def test_diag_broyden(self, capsys: CaptureFixture, pipe_velocity_flow: ConfigParser) -> None:
        pipe_velocity_flow['SOLVER']['nonlinear_solver'] = 'DiagBroyden'
        automated_output_check(capsys, pipe_velocity_flow, EXPECTED_ERRORS)
