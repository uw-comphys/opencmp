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

from pytest import CaptureFixture, fixture
from opencmp.helpers.testing import automated_output_check
from opencmp.config_functions import ConfigParser

# TODO: Figure out how quickly the poisson model should reach SS.

@fixture
def pid_poisson_1d() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config file for a transient 1D poisson solve with a tuned PID
    controller.

    The domain is a coarse 2D square mesh, but the boundary conditions were chosen so that spatial variations exist
    only in the x-direction.

    The left boundary has a smoothed-step change from 10 to 9.
    The control variable is the value at the middle of the domain x=0.5.
    The manipulated variable is the right boundary.

    Returns
        The config parser loaded with the config file
    """
    return ConfigParser('pytests/full_system/control/pid_poisson_1d/config')


class TestPIDController:
    def test_explicit_euler_cg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    # TODO: dt needs to be very small
    # def test_explicit_euler_dg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
    #     # Change from CG to DG
    #     pid_poisson_1d['DG']['DG'] = 'True'
    #     # Change time step
    #     pid_poisson_1d['TRANSIENT']['dt'] = '0.0005'
    #     # Run
    #     automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    def test_implicit_euler_cg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Change scheme
        pid_poisson_1d['TRANSIENT']['scheme'] = 'implicit euler'
        # Change time step
        pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    def test_implicit_euler_dg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Change scheme
        pid_poisson_1d['TRANSIENT']['scheme'] = 'implicit euler'
        # Change time step
        pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
        # Change from CG to DG
        pid_poisson_1d['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    def test_crank_nicolson_cg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Change scheme
        pid_poisson_1d['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    # TODO: dt needs to be very small
    # def test_crank_nicolson_dg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
    #     # Change scheme
    #     pid_poisson_1d['TRANSIENT']['scheme'] = 'crank nicolson'
    #     # Change time step
    #     pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
    #     # Change from CG to DG
    #     pid_poisson_1d['DG']['DG'] = 'True'
    #     # Run
    #     automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    def test_adaptive_two_step_cg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Change scheme
        pid_poisson_1d['TRANSIENT']['scheme'] = 'adaptive two step'
        # Change time step
        pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    # TODO: time step keeps getting smaller and smaller. Why?
    # def test_adaptive_two_step_dg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
    #     # Change scheme
    #     pid_poisson_1d['TRANSIENT']['scheme'] = 'adaptive two step'
    #     # Change time step
    #     pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
    #     # Change from CG to DG
    #     pid_poisson_1d['DG']['DG'] = 'True'
    #     # Run
    #     automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    def test_adaptive_three_step_cg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Change scheme
        pid_poisson_1d['TRANSIENT']['scheme'] = 'adaptive three step'
        # Change time step
        pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])

    def test_adaptive_three_step_dg(self, capsys: CaptureFixture, pid_poisson_1d: ConfigParser) -> None:
        # Change scheme
        pid_poisson_1d['TRANSIENT']['scheme'] = 'adaptive three step'
        # Change time step
        pid_poisson_1d['TRANSIENT']['dt'] = '0.05'
        # Change from CG to DG
        pid_poisson_1d['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, pid_poisson_1d, [2e-14, 1.5e-14, 8e-13])
