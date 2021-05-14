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
from config_functions import ConfigParser


@fixture
def pipe_unstructured_stationary() -> ConfigParser:
    """
    Fixture to return a ConfigParser loaded with a base config for stationary solves in a pipe.

    Returns:
        The config parser loaded with the config
    """
    return ConfigParser('pytests/full_system/stokes/stationary_pipe/config')


class TestStationary:
    def test_stationary_cg(self, capsys: CaptureFixture, pipe_unstructured_stationary: ConfigParser) -> None:
        # Run
        automated_output_check(capsys, pipe_unstructured_stationary, [1e-10, 6e-12, 3e-11, 2e-12, 6e-10, 2e-11, 3e-10])

    def test_stationary_dg(self, capsys: CaptureFixture, pipe_unstructured_stationary: ConfigParser) -> None:
        # Change from CG to DG
        pipe_unstructured_stationary['DG']['DG'] = 'True'
        # Change function spaces
        pipe_unstructured_stationary['FINITE ELEMENT SPACE']['elements'] = 'u -> HDiv\np -> L2'
        # Run
        automated_output_check(capsys, pipe_unstructured_stationary, [1e-10, 6e-12, 3e-11, 2e-12, 1e-11, 2e-11, 3e-10])
