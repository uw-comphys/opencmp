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
import pytest
from pytest import CaptureFixture, fixture
from opencmp.helpers.testing import automated_output_check
from opencmp.config_functions import ConfigParser

# TODO: https://stackoverflow.com/questions/45603930/pytest-parameterize-only-certain-permutations


@fixture
def diffusion() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient diffusion solves
    on a coarse unit square mesh.

    """
    return ConfigParser('pytests/full_system/mcins/diffusion/config')


@fixture
def diffusion_convection() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient diffusion and convection solves
    on a coarse unit square mesh.

    """
    return ConfigParser('pytests/full_system/mcins/diffusion_convection/config')


@fixture
def zero_order_rxn() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient 0th order + diffusion solves
    on a coarse unit square mesh.

    """
    return ConfigParser('pytests/full_system/mcins/0th_rxn/config')


@fixture
def first_order_rxn() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient 1st order uncoupled rxn + diffusion solves
    on a coarse unit square mesh.

    """
    return ConfigParser('pytests/full_system/mcins/1st_rxn/config')


@fixture
def first_order_rxn_coupled() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for transient 1st order coupled rxn + diffusion solves
    on a coarse unit square mesh.

    """
    return ConfigParser('pytests/full_system/mcins/1st_rxn_coupled/config')


class TestDiffusion:
    # def test_stationary_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Turn it to stationary
    #     diffusion['TRANSIENT']['transient'] = 'False'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    # def test_stationary_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Turn it to stationary
    #     diffusion['TRANSIENT']['transient'] = 'False'
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    def test_implicit_euler_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
        # Run
        expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
        automated_output_check(capsys, diffusion, expected_errors)

    # def test_implicit_euler_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    def test_crank_nicolson_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
        # Change time discretization
        diffusion['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
        automated_output_check(capsys, diffusion, expected_errors)

    # def test_crank_nicolson_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Change time discretization
    #     diffusion['TRANSIENT']['scheme'] = 'crank nicolson'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    @pytest.mark.slow
    def test_adaptive_2_step_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
        # Change time discretization
        diffusion['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        expected_errors = [0.0, 0.0, 8e-8, 8e-8, 3e-16, 0]
        automated_output_check(capsys, diffusion, expected_errors)

    # def test_adaptive_2_step_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Change time discretization
    #     diffusion['TRANSIENT']['scheme'] = 'adaptive two step'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    @pytest.mark.slow
    def test_adaptive_3_step_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
       # Change time discretization
       diffusion['TRANSIENT']['scheme'] = 'adaptive three step'
       # Run
       expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
       automated_output_check(capsys, diffusion, expected_errors)

    # def test_adaptive_3_step_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Change time discretization
    #     diffusion['TRANSIENT']['scheme'] = 'adaptive three step'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    def test_imex_euler_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
        # Change liniearization scheme
        diffusion['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        diffusion['TRANSIENT']['scheme'] = 'euler IMEX'
        # Run
        expected_errors = [0.0, 0.0, 4e-9, 4e-9, 3e-16, 0]
        automated_output_check(capsys, diffusion, expected_errors)

    # def test_imex_euler_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Change liniearization scheme
    #     diffusion['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization
    #     diffusion['TRANSIENT']['scheme'] = 'euler IMEX'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    def test_imex_cnlf_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
        # Change liniearization scheme
        diffusion['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        diffusion['TRANSIENT']['scheme'] = 'CNLF'
        # Run
        expected_errors = [0.0, 0.0, 3e-9, 3e-9, 2.5e-16, 0]
        automated_output_check(capsys, diffusion, expected_errors)

    # def test_imex_cnlf_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change liniearization scheme
    #     diffusion['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change time discretization
    #     diffusion['TRANSIENT']['scheme'] = 'CNLF'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)

    def test_imex_sbdf_cg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
        # Change liniearization scheme
        diffusion['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        diffusion['TRANSIENT']['scheme'] = 'SBDF'
        # Run
        expected_errors = [0.0, 0.0, 3e-9, 3e-9, 5e-16, 0]
        automated_output_check(capsys, diffusion, expected_errors)

    # def test_imex_sbdf_dg(self, capsys: CaptureFixture, diffusion: ConfigParser) -> None:
    #     # Change liniearization scheme
    #     diffusion['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1\nc -> H1'
    #     # Change from CG to DG
    #     diffusion['DG']['DG'] = 'True'
    #     # Change time discretization
    #     diffusion['TRANSIENT']['scheme'] = 'CNLF'
    #     # Run
    #     expected_errors = [0.0, 0.0, 5e-8, 5e-8, 3e-16, 0]
    #     automated_output_check(capsys, diffusion, expected_errors)


class TestDiffusionConvection:
    # @pytest.mark.slow
    def test_implicit_euler_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
        # Run
        expected_errors = [3e-8, 1e-6, 4e-8, 2e-7]
        automated_output_check(capsys, diffusion_convection, expected_errors)

    # @pytest.mark.slow
    # def test_implicit_euler_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1
    #     # Run
    #     expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)

    # @pytest.mark.slow
    def test_crank_nicolson_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
        # Change time discretization
        diffusion_convection['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        expected_errors = [8e-9, 1e-6, 2e-8, 6e-8]
        automated_output_check(capsys, diffusion_convection, expected_errors)

    # @pytest.mark.slow
    # def test_crank_nicolson_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change time discretization
    #     diffusion_convection['TRANSIENT']['scheme'] = 'crank nicolson'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Run
    #     expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)

    # @pytest.mark.slow
    def test_adaptive_2_step_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
        # Change time discretization
        diffusion_convection['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        expected_errors = [8e-9, 1e-6, 2e-8, 6e-8]
        automated_output_check(capsys, diffusion_convection, expected_errors)

    # def test_adaptive_2_step_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change time discretization
    #     diffusion_convection['TRANSIENT']['scheme'] = 'adaptive two step'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Run
    #     expected_errors = [8e-9, 1e-7, 2e-8, 6e-8]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)

    # @pytest.mark.slow
    def test_adaptive_3_step_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
       # Change time discretization
       diffusion_convection['TRANSIENT']['scheme'] = 'adaptive three step'
       # Change timestepping tolerances
       diffusion_convection['TRANSIENT']['dt_tolerance'] = 'relative -> 1e-6 \n absolute -> 6e-4'
       # Run
       expected_errors = [8e-9, 1e-6, 2e-8, 6e-8]
       automated_output_check(capsys, diffusion_convection, expected_errors)

    # def test_adaptive_3_step_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change time discretization
    #     diffusion_convection['TRANSIENT']['scheme'] = 'adaptive three step'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Run
    #     expected_errors = [8e-9, 1e-7, 2e-8, 6e-8]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)

    #def test_imex_euler_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #    # Change liniearization scheme
    #    diffusion_convection['SOLVER']['linearization_method'] = 'IMEX'
    #    # Change time discretization
    #    diffusion_convection['TRANSIENT']['scheme'] = 'euler IMEX'
    #    # Run
    #    expected_errors = [5e-16, 1.4e-7, 1e-7, 8e-7]
    #    automated_output_check(capsys, diffusion_convection, expected_errors)

    # def test_imex_euler_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change liniearization scheme
    #     diffusion_convection['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization
    #     diffusion_convection['TRANSIENT']['scheme'] = 'euler IMEX'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Run
    #     expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)

    #def test_imex_cnlf_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #    # Change liniearization scheme
    #    diffusion_convection['SOLVER']['linearization_method'] = 'IMEX'
    #    # Change time discretization
    #    diffusion_convection['TRANSIENT']['scheme'] = 'CNLF'
    #    # Run
    #    expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #    automated_output_check(capsys, diffusion_convection, expected_errors)

    # def test_imex_cnlf_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change liniearization scheme
    #     diffusion_convection['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization
    #     diffusion_convection['TRANSIENT']['scheme'] = 'CNLF'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1'
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Run
    #     expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)

    #def test_imex_sbdf_cg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #    # Change liniearization scheme
    #    diffusion_convection['SOLVER']['linearization_method'] = 'IMEX'
    #    # Change time discretization
    #    diffusion_convection['TRANSIENT']['scheme'] = 'SBDF'
    #    # Run
    #    expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #    automated_output_check(capsys, diffusion_convection, expected_errors)

    # def test_imex_sbdf_dg(self, capsys: CaptureFixture, diffusion_convection: ConfigParser) -> None:
    #     # Change liniearization scheme
    #     diffusion_convection['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization
    #     diffusion_convection['TRANSIENT']['scheme'] = 'SBDF'
    #     # Change elements
    #     diffusion['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2\na -> H1\nb -> H1
    #     # Change from CG to DG
    #     diffusion_convection['DG']['DG'] = 'True'
    #     # Run
    #     expected_errors = [8.7e-8, 5e-6, 1e-7, 8e-7]
    #     automated_output_check(capsys, diffusion_convection, expected_errors)


class TestZeroOrderRxn:
    def test_implicit_euler_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
        # Run
        expected_errors = [0.0, 7e-15, 1e-14, 0.0]
        automated_output_check(capsys, zero_order_rxn, expected_errors)

    def test_crank_nicolson_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
        # Change time discretization
        zero_order_rxn['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        expected_errors = [0.0, 1e-14, 2e-15, 0.0]
        automated_output_check(capsys, zero_order_rxn, expected_errors)

    def test_adaptive_2_step_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
        # Change time discretization
        zero_order_rxn['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        expected_errors = [0.0, 2e-15, 4e-15, 0.0]
        automated_output_check(capsys, zero_order_rxn, expected_errors)

    def test_adaptive_3_step_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
       # Change time discretization
       zero_order_rxn['TRANSIENT']['scheme'] = 'adaptive three step'
       # Run
       expected_errors = [0.0, 2e-15, 3.5e-15, 0.0]
       automated_output_check(capsys, zero_order_rxn, expected_errors)

    def test_imex_euler_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
        # Change liniearization scheme
        zero_order_rxn['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        zero_order_rxn['TRANSIENT']['scheme'] = 'euler IMEX'
        # Run
        expected_errors = [0.0, 7.5e-15, 1e-14, 0.0]
        automated_output_check(capsys, zero_order_rxn, expected_errors)

    # TODO: Doesn't seem to be stable.
    # def test_imex_cnlf_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
    #     # Change liniearization scheme
    #     zero_order_rxn['SOLVER']['linearization_method'] = 'IMEX'
    #     # Change time discretization
    #     zero_order_rxn['TRANSIENT']['scheme'] = 'CNLF'
    #     # Run
    #     expected_errors = [0.0, 1e-11, 1e-11, 0.0]
    #     automated_output_check(capsys, zero_order_rxn, expected_errors)

    # TODO: Doesn't seem to be stable.
    #def test_imex_sbdf_cg(self, capsys: CaptureFixture, zero_order_rxn: ConfigParser) -> None:
    #    # Change liniearization scheme
    #    zero_order_rxn['SOLVER']['linearization_method'] = 'IMEX'
    #    # Change time discretization
    #    zero_order_rxn['TRANSIENT']['scheme'] = 'SBDF'
    #    # Run
    #    expected_errors = [0.0, 2e-15, 3.5e-15, 0.0]
    #    automated_output_check(capsys, zero_order_rxn, expected_errors)


class TestFirstOrderRxn:
    def test_implicit_euler_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
        # Run
        expected_errors = [0.0, 1e-9, 1e-16, 0.0]
        automated_output_check(capsys, first_order_rxn, expected_errors)

    def test_crank_nicolson_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
        # Change time discretization
        first_order_rxn['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        expected_errors = [0.0, 5.5e-12, 1e-16, 0.0]
        automated_output_check(capsys, first_order_rxn, expected_errors)

    def test_adaptive_2_step_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
        # Change time discretization
        first_order_rxn['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        expected_errors = [0.0, 5e-6, 1e-16, 0.0]
        automated_output_check(capsys, first_order_rxn, expected_errors)

    def test_adaptive_3_step_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
       # Change time discretization
       first_order_rxn['TRANSIENT']['scheme'] = 'adaptive three step'
       # Run
       expected_errors = [0.0, 8e-7, 1e-16, 0.0]
       automated_output_check(capsys, first_order_rxn, expected_errors)

    def test_imex_euler_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
        # Change liniearization scheme
        first_order_rxn['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        first_order_rxn['TRANSIENT']['scheme'] = 'euler IMEX'
        # Run
        expected_errors = [0.0, 1e-9, 1e-16, 0.0]
        automated_output_check(capsys, first_order_rxn, expected_errors)

    # TODO: Find a stable time step
    #def test_imex_cnlf_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
    #    # Change liniearization scheme
    #    first_order_rxn['SOLVER']['linearization_method'] = 'IMEX'
    #    # Change time discretization
    #    first_order_rxn['TRANSIENT']['scheme'] = 'CNLF'
    #    # Run
    #    expected_errors = [0.0, 8e-7, 1.e-30, 0.0]
    #    automated_output_check(capsys, first_order_rxn, expected_errors)

    def test_imex_sbdf_cg(self, capsys: CaptureFixture, first_order_rxn: ConfigParser) -> None:
        # Change liniearization scheme
        first_order_rxn['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        first_order_rxn['TRANSIENT']['scheme'] = 'SBDF'
        # Run
        expected_errors = [0.0, 4e-11, 1e-16, 0.0]
        automated_output_check(capsys, first_order_rxn, expected_errors)


class TestFirstOrderRxnCoupled:
    def test_implicit_euler_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
        # Run
        expected_errors = [0.0, 9.9e-5, 8.9e-5, 9.9e-6, 0.0]
        automated_output_check(capsys, first_order_rxn_coupled, expected_errors)

    def test_crank_nicolson_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
        # Change time discretization
        first_order_rxn_coupled['TRANSIENT']['scheme'] = 'crank nicolson'
        # Run
        expected_errors = [0.0, 1.5e-7, 1.6e-7, 7e-9, 0.0]
        automated_output_check(capsys, first_order_rxn_coupled, expected_errors)

    @pytest.mark.slow
    def test_adaptive_2_step_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
        # Change time discretization
        first_order_rxn_coupled['TRANSIENT']['scheme'] = 'adaptive two step'
        # Run
        expected_errors = [0.0, 7.6e-5, 7e-5, 8e-6, 0.0]
        automated_output_check(capsys, first_order_rxn_coupled, expected_errors)

    @pytest.mark.slow
    def test_adaptive_3_step_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
       # Change time discretization
       first_order_rxn_coupled['TRANSIENT']['scheme'] = 'adaptive three step'
       # Run
       expected_errors = [0.0, 9.9e-5, 8.9e-5, 9.9e-6, 0.0]
       automated_output_check(capsys, first_order_rxn_coupled, expected_errors)

    def test_imex_euler_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
        # Change liniearization scheme
        first_order_rxn_coupled['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        first_order_rxn_coupled['TRANSIENT']['scheme'] = 'euler IMEX'
        # Run
        expected_errors = [0.0, 9.9e-5, 8.9e-5, 9.9e-6, 0.0]
        automated_output_check(capsys, first_order_rxn_coupled, expected_errors)

    # TODO: Find a stable time step.
    #def test_imex_cnlf_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
    #    # Change liniearization scheme
    #    first_order_rxn_coupled['SOLVER']['linearization_method'] = 'IMEX'
    #    # Change time discretization
    #    first_order_rxn_coupled['TRANSIENT']['scheme'] = 'CNLF'
    #    # Run
    #    expected_errors = [0.0, 9.9e-5, 8.9e-5, 9.9e-6, 0.0]
    #    automated_output_check(capsys, first_order_rxn_coupled, expected_errors)

    def test_imex_sbdf_cg(self, capsys: CaptureFixture, first_order_rxn_coupled: ConfigParser) -> None:
        # Change liniearization scheme
        first_order_rxn_coupled['SOLVER']['linearization_method'] = 'IMEX'
        # Change time discretization
        first_order_rxn_coupled['TRANSIENT']['scheme'] = 'SBDF'
        # Run
        expected_errors = [0.0, 7.5e-5, 4e-5, 1e-4, 0.0]
        automated_output_check(capsys, first_order_rxn_coupled, expected_errors)
