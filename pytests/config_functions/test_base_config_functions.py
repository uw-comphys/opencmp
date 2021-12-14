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
from opencmp.config_functions import ConfigParser,ConfigFunctions
import os
import ngsolve as ngs


class TestInitialization:
    """ Class to test initializing a ConfigFunction object. """

    def test_1(self):
        """ Check that run_dir was parsed correctly from the config file path. """
        test_config_functions = ConfigFunctions('pytests/config_functions/example_config', 'import_functions.py', None,
                                                ngs.Parameter(1.5))

        assert test_config_functions.run_dir == 'pytests/config_functions'

    def test_2(self):
        """ Check that just the name of a file can be given if run_dir is the current directory. """
        # Need to change the working directory to get this to work.
        cur_dir = os.getcwd()
        os.chdir('pytests/config_functions')
        test_config_functions = ConfigFunctions('example_config', 'import_functions.py', None, ngs.Parameter(1.5))

        assert test_config_functions.run_dir == '.'

        # Switch back to the original directory so other tests don't get screwed up.
        os.chdir(cur_dir)

    def test_3(self):
        """ Check that a config parser object was actually created and contains all of the expected values. """
        test_config_functions = ConfigFunctions('pytests/config_functions/example_config', 'import_functions.py', None,
                                                ngs.Parameter(1.5))

        expected_config = ConfigParser('pytests/config_functions/example_config')
        
        assert test_config_functions.config == expected_config

    def test_4(self):
        """ Check that an error is raised if the config file does not exist. """
        with pytest.raises(FileNotFoundError):
            test_config_functions = ConfigFunctions('pytests/config_functions/missing_config', 'import_functions.py',
                                                    None, ngs.Parameter(1.5))

    def test_5(self):
        """ Check that t_param was set correctly. """
        test_config_functions = ConfigFunctions('pytests/config_functions/example_config', 'import_functions.py', None,
                                                ngs.Parameter(1.5))

        assert isinstance(test_config_functions.t_param, ngs.Parameter)
        assert test_config_functions.t_param.Get() == 1.5

class TestFindRelPathForFile:
    """ Class to test ConfigFunction._find_rel_path_for_file. """
    def test_1(self):
        """ Check that a file in the current working directory can be found. """
        # Need to change the working directory to get this to work.
        cur_dir = os.getcwd()
        os.chdir('pytests/config_functions')
        test_config_functions = ConfigFunctions('example_config', 'import_functions.py', None, ngs.Parameter(0.0))

        # Confirm that the path to the file is correct.
        rel_path = test_config_functions._find_rel_path_for_file('example_config')
        assert rel_path == 'example_config'

        # Switch back to the original directory so other tests don't get screwed up.
        os.chdir(cur_dir)

    def test_2(self):
        """ Check that a file in the specified run directory can be found. """
        test_config_functions = ConfigFunctions('pytests/config_functions/example_config', 'import_functions.py', None,
                                                ngs.Parameter(0.0))

        # Confirm that the path to the file is correct.
        rel_path = test_config_functions._find_rel_path_for_file('example_config')
        assert rel_path == 'pytests/config_functions/example_config'

    def test_3(self):
        """
        Check that a file in the main run directory (one level up from the specified run directory) can be found.
        """
        test_config_functions = ConfigFunctions('pytests/config_functions/example_config', 'import_functions.py', None,
                                                ngs.Parameter(0.0))

        # Confirm that the path to the file is correct.
        rel_path = test_config_functions._find_rel_path_for_file('conftest.py')
        assert rel_path == 'pytests/config_functions/../conftest.py'

    def test_4(self):
        """ Check that a file that can't be found raises an error. """
        test_config_functions = ConfigFunctions('pytests/config_functions/example_config', 'import_functions.py', None,
                                                ngs.Parameter(0.0))

        with pytest.raises(FileNotFoundError):
            rel_path = test_config_functions._find_rel_path_for_file('missing_config')

class TestReParse:
    """ Class to test ConfigFunction.re_parse. """
    # TODO:
