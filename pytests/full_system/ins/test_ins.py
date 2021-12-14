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
from pytests.full_system.helpers import automated_output_check
from opencmp.config_functions import ConfigParser


@fixture
def pipe_velocity_flow() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for stationary flow in a pipe with velocity BCs.

    Returns:
        The config parser loaded with the config
    """
    return ConfigParser('pytests/full_system/ins/pressure_flow_in_pipe_velocity/config')


# TODO: Once new BCs are working
@fixture
def pipe_stress_flow() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for stationary flow in a pipe with total stress BCs.

    Returns:
        The config parser loaded with the config
    """
    return ConfigParser('pytests/full_system/ins/pressure_flow_in_pipe_stress/config')


@fixture
def sinusoidal_transient() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient flow in a square mesh with sinusoidal,
    time varying BCs.

    Returns:
        The config parser loaded with the config
    """
    return ConfigParser('pytests/full_system/ins/sinusoidal_transient/config')


class TestStationary:
    def test_pipe_flow_velocity_cg(self, capsys: CaptureFixture, pipe_velocity_flow: ConfigParser) -> None:
        # Run
        automated_output_check(capsys, pipe_velocity_flow, [2e-7, 5e-8, 1.5e-4, 3.5e-5])

    def test_pipe_flow_velocity_dg(self, capsys: CaptureFixture, pipe_velocity_flow: ConfigParser) -> None:
        # Change from CG to DG
        pipe_velocity_flow['DG']['DG'] = 'True'
        # Change elements
        pipe_velocity_flow['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, pipe_velocity_flow, [2e-7, 5e-8, 4.5e-7, 2e-12])


class TestTransient:
    def test_sinusoidal_oseen_implicit_euler_cg(self, capsys: CaptureFixture,
                                                sinusoidal_transient: ConfigParser) -> None:
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])

    def test_sinusoidal_oseen_implicit_euler_dg(self, capsys: CaptureFixture,
                                                sinusoidal_transient: ConfigParser) -> None:
        # Change from CG to DG
        sinusoidal_transient['DG']['DG'] = 'True'
        # Change elements
        sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 5e-11])

    def test_sinusoidal_oseen_crank_nicolson_cg(self, capsys: CaptureFixture,
                                                sinusoidal_transient: ConfigParser) -> None:
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])

    def test_sinusoidal_oseen_crank_nicolson_dg(self, capsys: CaptureFixture,
                                                sinusoidal_transient: ConfigParser) -> None:
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'crank nicolson'
        # Change from CG to DG
        sinusoidal_transient['DG']['DG'] = 'True'
        # Change elements
        sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [2.5e-3, 1e-3])

    def test_sinusoidal_oseen_adaptive_two_step_cg(self, capsys: CaptureFixture,
                                                   sinusoidal_transient: ConfigParser) -> None:
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])

    def test_sinusoidal_oseen_adaptive_two_step_dg(self, capsys: CaptureFixture,
                                                   sinusoidal_transient: ConfigParser) -> None:
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'adaptive two step'
        # Change from CG to DG
        sinusoidal_transient['DG']['DG'] = 'True'
        # Change elements
        sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [2.5e-3, 1e-3])

    #def test_sinusoidal_oseen_adaptive_three_step_cg(self, capsys: CaptureFixture,
    #                                                sinusoidal_transient: ConfigParser) -> None:
    #   # Change time discretization scheme
    #   sinusoidal_transient['TRANSIENT']['scheme'] = 'adaptive three step'
    #   # Run
    #   automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])
    #
    #def test_sinusoidal_oseen_adaptive_three_step_dg(self, capsys: CaptureFixture,
    #                                                sinusoidal_transient: ConfigParser) -> None:
    #   # Change time discretization scheme
    #   sinusoidal_transient['TRANSIENT']['scheme'] = 'adaptive three step'
    #   # Change from CG to DG
    #   sinusoidal_transient['DG']['DG'] = 'True'
    #   # Change elements
    #   sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
    #   # Run
    #   automated_output_check(capsys, sinusoidal_transient, [2.5e-3, 1e-3])

    def test_sinusoidal_imex_euler_cg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
        # Change linearization scheme
        sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'euler IMEX'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])

    def test_sinusoidal_imex_euler_dg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
        # Change linearization scheme
        sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'euler IMEX'
        # Change from CG to DG
        sinusoidal_transient['DG']['DG'] = 'True'
        # Change elements
        sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 5e-11])

    def test_sinusoidal_imex_cnlf_cg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
        # Change linearization scheme
        sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'CNLF'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])

    def test_sinusoidal_imex_cnlf_dg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
        # Change linearization scheme
        sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'CNLF'
        # Change from CG to DG
        sinusoidal_transient['DG']['DG'] = 'True'
        # Change elements
        sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [2.5e-3, 1e-3])

    def test_sinusoidal_imex_sbdf_cg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
        # Change linearization scheme
        sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'SBDF'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])

    def test_sinusoidal_imex_sbdf_dg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
        # Change linearization scheme
        sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization scheme
        sinusoidal_transient['TRANSIENT']['scheme'] = 'SBDF'
        # Change from CG to DG
        sinusoidal_transient['DG']['DG'] = 'True'
        # Change elements
        sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, sinusoidal_transient, [1e-4, 5e-11])

    # TODO: Turn on once RK 222 and RK 232 are working properly
    # def test_sinusoidal_imex_rk_222_cg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
    #     # Change linearization scheme
    #     sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization scheme
    #     sinusoidal_transient['TRANSIENT']['scheme'] = 'RK 222'
    #     # Run
    #     automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])
    #
    # def test_sinusoidal_imex_rk_222_dg(self, capsys: CaptureFixture,
    #                                             sinusoidal_transient: ConfigParser) -> None:
    #     # Change linearization scheme
    #     sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization scheme
    #     sinusoidal_transient['TRANSIENT']['scheme'] = 'RK 222'
    #     # Change from CG to DG
    #     sinusoidal_transient['DG']['DG'] = 'True'
    #     # Change elements
    #     sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
    #     # Run
    #     automated_output_check(capsys, sinusoidal_transient, [1e-4, 5e-11])
    #
    # def test_sinusoidal_imex_rk_232_cg(self, capsys: CaptureFixture, sinusoidal_transient: ConfigParser) -> None:
    #     # Change linearization scheme
    #     sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization scheme
    #     sinusoidal_transient['TRANSIENT']['scheme'] = 'RK 232'
    #     # Run
    #     automated_output_check(capsys, sinusoidal_transient, [1e-4, 2e-3])
    #
    # def test_sinusoidal_imex_rk_232_dg(self, capsys: CaptureFixture,
    #                                             sinusoidal_transient: ConfigParser) -> None:
    #     # Change linearization scheme
    #     sinusoidal_transient['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization scheme
    #     sinusoidal_transient['TRANSIENT']['scheme'] = 'RK 232'
    #     # Change from CG to DG
    #     sinusoidal_transient['DG']['DG'] = 'True'
    #     # Change elements
    #     sinusoidal_transient['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
    #     # Run
    #     automated_output_check(capsys, sinusoidal_transient, [1e-4, 5e-11])
