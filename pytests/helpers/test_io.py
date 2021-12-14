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
from opencmp.config_functions import ConfigParser
import opencmp.helpers.io as io
from ngsolve import Mesh


@fixture
def empty_config() -> ConfigParser:
    """
    Function to return a config parser initialized with an empty config file.

    Returns:
        Config parser initialized with empty config file.
    """
    return ConfigParser('pytests/helpers/mesh/config_blank')


class TestCreateAndLoadGridfunctionFromFile:
    def test_bad_filename(self, capsys: CaptureFixture):
        try:
            io.create_and_load_gridfunction_from_file("", None)
        except FileNotFoundError:
            # Working as expected
            pass


class TestLoadMesh:
    def test_invalid_type(self, empty_config: ConfigParser):
        """
        Test with filename having an invalid type.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        # Add required options to empty config file
        empty_config['MESH'] = {'filename': 'pytests/helpers/mesh/config_blank'}

        try:
            io.load_mesh(empty_config)
        except TypeError:
            # Working as intended
            pass

    def test_no_filename(self, empty_config: ConfigParser):
        """
        Test with no filename in the config file.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        try:
            io.load_mesh(empty_config)
        except ValueError:
            # Working as intended
            pass

    def test_file_does_not_exist(self, empty_config: ConfigParser):
        """
        Test with a filename for a file that doesn't exist.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        # Add required options to empty config file
        empty_config['MESH'] = {'filename': 'pytests/mesh_files/channel_3bcs.vol'}

        try:
            io.load_mesh(empty_config)
        except FileNotFoundError:
            # Working as intended
            pass

    def test_valid_vol(self, empty_config: ConfigParser):
        """
        Test with a valid .vol type mesh.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        # Add required options to empty config file
        empty_config['MESH'] = {'filename': 'pytests/mesh_files/channel_3bcs.vol'}

        # Load mesh
        mesh = io.load_mesh(empty_config)

        # Check various properties that we know the mesh should have
        assert type(mesh) is Mesh
        assert mesh.dim == 2
        for boundary in mesh.GetBoundaries():
            assert boundary in ('wall', 'outlet', 'inlet')
        assert mesh.nedge == 1137
        assert mesh.nface == 726
        assert mesh.nfacet == 1137

    def test_valid_msh(self, empty_config: ConfigParser):
        """
        Test with a valid .msh type mesh.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        # Add required options to empty config file
        empty_config['MESH'] = {'filename': 'pytests/mesh_files/square.msh'}

        mesh = io.load_mesh(empty_config)

        assert type(mesh) is Mesh
        assert mesh.dim == 2
        for boundary in mesh.GetBoundaries():
            assert boundary in ('left', 'right', 'top', 'bottom')
        assert mesh.nedge == 259
        assert mesh.nface == 162
        assert mesh.nfacet == 259

    def test_curved_element_vol(self, empty_config: ConfigParser):
        """
        Test with a valid .vol mesh which is supposed to have curved elements.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        # Test curve = 1
        empty_config['MESH'] = {'filename': 'pytests/mesh_files/channel_3bcs.vol',
                                'curved_elements': 'True'}
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '1'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 1

        # Test curve = 2
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '2'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 2

        # Test curve = 3
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '3'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 3

        # Test curve = 50
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '50'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 50

    def test_curved_element_msh(self, empty_config: ConfigParser):
        """
        Test with a valid .msh mesh which is supposed to have curved elements.

        Args:
            empty_config: Config parser initialized with empty config file.
        """
        # Test curve = 1
        empty_config['MESH'] = {'filename': 'pytests/mesh_files/square.msh',
                                'curved_elements': 'True'}
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '1'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 1

        # Test curve = 2
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '2'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 2

        # Test curve = 3
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '3'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 3

        # Test curve = 50
        empty_config['FINITE ELEMENT SPACE'] = {'interpolant_order': '50'}
        mesh = io.load_mesh(empty_config)
        assert mesh.GetCurveOrder() == 50

# class UpdateGridFunctionFromFile:
#     pass
