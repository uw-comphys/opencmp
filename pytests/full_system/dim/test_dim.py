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

from pytests.full_system.helpers import run_example
from config_functions import ConfigParser


class TestStationary:
    def test_DIM_poisson_1(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_poisson_1/config'))

    def test_DIM_poisson_2(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_poisson_2/config'))

    def test_DIM_poisson_3(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_poisson_3/config'))

    def test_DIM_poisson_4(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_poisson_4/config'))

    def test_DIM_poisson_5(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_poisson_5/config'))

    def test_DIM_poisson_6(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_poisson_6/config'))

    def test_DIM_stokes_1(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_stokes_1/config'))

    def test_DIM_stokes_2(self) -> None:
        run_example(ConfigParser('pytests/full_system/dim/dim_stokes_2/config'))
