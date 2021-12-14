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

from opencmp.helpers.misc import merge_bc_dict


class TestMergeBCDict:
    """
    Test the implementation of the BC dictionary merging helper function.
    """
    def test_empty_empty(self):
        res = merge_bc_dict({}, {})

        assert len(res) == 0

    def test_1_layer_non_overlapping_keys(self):
        d1      = {1: [1]}
        d2      = {2: [2]}
        correct = {1: [1],
                   2: [2]}

        d1 = merge_bc_dict(d1, d2)

        assert d1 == correct

    def test_1_layer_overlapping_keys_non_overlapping_vals(self):
        d1      = {1: [1, None]}
        d2      = {1: [None, 2]}
        correct = {1: [1, 2]}

        d1 = merge_bc_dict(d1, d2)

        assert d1 == correct

    def test_1_layer_overlapping_keys_overlapping_vals(self):
        d1      = {1: [1, 3]}
        d2      = {1: [None, 2]}
        correct = {1: [1, 2]}

        d1 = merge_bc_dict(d1, d2)

        assert d1 == correct

    def test_2_layer_non_overlapping_keys(self):
        d1      = {1: {1: [1]}}
        d2      = {2: {2: [2]}}
        correct = {1: {1: [1]},
                   2: {2: [2]}}

        d1 = merge_bc_dict(d1, d2)

        assert d1 == correct

    def test_2_layer_overlapping_keys(self):
        d1      = {1: {1: [1],
                       2: [2]},
                   3: {3: [3]}}
        d2      = {1: {4: [4]},
                   3: {3: [5]}}
        correct = {1: {1: [1],
                       2: [2],
                       4: [4]},
                   3: {3: [5]}}

        d1 = merge_bc_dict(d1, d2)

        assert d1 == correct
