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
from pytests.full_system.helpers import automated_output_check, manual_output_check
from opencmp.config_functions import ConfigParser


@fixture
def square_coarse_h_converge() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for an h (mesh element size) convergence test.

    Returns:
        The config parser loaded with the config
    """
    return ConfigParser('pytests/full_system/poisson/h_convergence/config')


@fixture
def square_coarse_transient() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient solves on a coarse unit square mesh.

    Returns:
        The config parser loaded with the config
    """
    return ConfigParser('pytests/full_system/poisson/transient_coarse/config')


class TestStationary:
    def test_h_convergence_cg(self, capsys: CaptureFixture, square_coarse_h_converge: ConfigParser) -> None:
        # Run
        manual_output_check(capsys, 'h convergence CG', square_coarse_h_converge)

    def test_h_convergence_dg(self, capsys: CaptureFixture, square_coarse_h_converge: ConfigParser) -> None:
        # Change from CG to DG
        square_coarse_h_converge['DG']['DG'] = 'True'
        # Run
        manual_output_check(capsys, 'h convergence DG', square_coarse_h_converge)


class TestTransient:
    def test_explicit_euler_cg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_explicit_euler_dg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change from CG to DG
        square_coarse_transient['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_implicit_euler_cg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'implicit euler'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_implicit_euler_dg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'implicit euler'
        # Change from CG to DG
        square_coarse_transient['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_crank_nicolson_cg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_crank_nicolson_dg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'crank nicolson'
        # Change from CG to DG
        square_coarse_transient['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_2_step_adaptive_cg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_2_step_adaptive_dg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'adaptive two step'
        # Change from CG to DG
        square_coarse_transient['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_3_step_adaptive_cg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'adaptive three step'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])

    def test_3_step_adaptive_dg(self, capsys: CaptureFixture, square_coarse_transient: ConfigParser) -> None:
        # Change scheme
        square_coarse_transient['TRANSIENT']['scheme'] = 'adaptive three step'
        # Change from CG to DG
        square_coarse_transient['DG']['DG'] = 'True'
        # Run
        automated_output_check(capsys, square_coarse_transient, [9e-6])
