"""
Copyright 2021 the authors (see AUTHORS file for full list)

This file is part of OpenCMP.

OpenCMP is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

OpenCMP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with OpenCMP.  If not, see <https://www.gnu.org/licenses/>.
"""

from config_functions import ConfigParser
from config_functions import load_config
from . import interface, mesh_helpers
from helpers.ngsolve_ import numpy_to_NGSolve, gridfunction_rigid_body_motion
from helpers.io import load_mesh, create_and_load_gridfunction_from_file
import numpy as np
import ngsolve as ngs


class DIM:
    """
    Class to hold the diffuse interface methods and parameters.
    """

    def __init__(self, DIM_dir: str) -> None:
        """
        Initialize the diffuse interface parameters.

        Args:
            DIM_dir: The run directory for the diffuse interface parameters.
        """
        # Save the path to the DIM directory.
        self.DIM_dir = DIM_dir

        # Load the DIM-specific configfile.
        self.config = ConfigParser(self.DIM_dir + '/config')

        # Load the BC parameters.
        self.multiple_bcs = self.config.get_item(['DIM BOUNDARY CONDITIONS', 'multiple_bcs'], bool)

        # Initialize the mask dictionaries.
        self.mask_arr_dict = {}
        self.mask_gfu_dict = {}

        # Determine if the phase fields and masks should be loaded from files or generated.
        self.load_method = self.config.get_item(['PHASE FIELDS', 'load_method'], str)
        if self.load_method == 'generate':
            # Generate from one .stl file.
            #
            # Load the nonconformal parameters.
            self._load_nonconformal_parameters()

            # Get the name of the .stl file holding the complex geometry.
            self.stl_filename = self.config.get_item(['PHASE FIELDS', 'stl_filename'], str)

            # Generate the phase field array and the nonconformal mesh.
            self.phi_arr = self._generate_phase_field()
            self._generate_DIM_mesh()

            # Generate the BC masks.
            if self.multiple_bcs:
                self.bc_config = ConfigParser(self.DIM_dir + '/bc_dir/config')
                self._load_bc_parameters(self.config, self.bc_config)
                self._generate_BC_masks()

        elif self.load_method == 'combine':
            # Generate phase fields from multiple .stl files and combine them into a single one.
            #
            # Load the nonconformal parameters.
            self._load_nonconformal_parameters()

            # Create the mesh.
            self._generate_DIM_mesh()

            # Create an array to hold the final phi.
            if self.invert:
                self.phi_arr = np.ones(tuple([n+1 for n in self.N]))
            else:
                self.phi_arr = np.zeros(tuple([n + 1 for n in self.N]))

            # Get the names of the .stl files and the config files holding further information about them.
            stl_filename_dict, _ = load_config.convert_str_to_dict(self.config.load_param_simple(
                ['PHASE FIELDS', 'stl_filename'], str), 0.0, {}, ['.stl'])
            for config_filename, stl_filename in stl_filename_dict.items():
                # Generate a phase field from the .stl file and incorporate it into self.phi_arr by taking the
                # elementwise maximum.
                self.stl_filename = stl_filename
                tmp_phi_arr = self._generate_phase_field()
                if self.invert:
                    self.phi_arr = np.minimum(self.phi_arr, tmp_phi_arr)
                else:
                    self.phi_arr = np.maximum(self.phi_arr, tmp_phi_arr)

                tmp_config = ConfigParser(self.DIM_dir + '/' + config_filename)
                tmp_multiple_bcs = tmp_config.get_item(['DIM BOUNDARY CONDITIONS', 'multiple_bcs'], bool)
                if tmp_multiple_bcs:
                    # Load BC parameters specific to the new config file.
                    self._load_bc_parameters(tmp_config, tmp_config)

                    # Add additional masks to self.mask_arr_dict based on the .stl file's config file.
                    self._generate_BC_masks()
                else:
                    # One mask of all ones. Name it after the config file.
                    mask = np.ones(tuple([n + 1 for n in self.tmp_N]))
                    mask = mesh_helpers.crop_to_mesh_bounds(mask, self.N, self.scale, self.offset, self.tmp_N,
                                                            self.tmp_scale, self.tmp_offset)
                    self.mask_arr_dict[config_filename] = mask

            # Reset the BC parameters back to the values from the main DIM config file.
            if self.multiple_bcs:
                self.bc_config = ConfigParser(self.DIM_dir + '/bc_dir/config')
                self._load_bc_parameters(self.config, self.bc_config)

        elif self.load_method == 'file':
            # Load from a .sol file.
            #
            if self.multiple_bcs:
                # Load BC parameters.
                self.bc_config = ConfigParser(self.DIM_dir + '/bc_dir/config')
                self._load_bc_parameters(self.config, self.bc_config, quiet=True)

        else:
            raise ValueError('Do not recognize method for loading the phase fields.')

        # Check for rigid body motion.
        if 'RIGID BODY MOTION' in self.config.sections():
            self.rigid_body_motion = True
            theta = load_config.parse_str(self.config.load_param_simple(['RIGID BODY MOTION', 'rotation_speed']), 0.0)
            self.b = np.array(load_config.parse_str(self.config.load_param_simple(['RIGID BODY MOTION', 'center_of_rotation']), 0.0)).transpose()
            speed = load_config.parse_str(self.config.load_param_simple(['RIGID BODY MOTION', 'translation_vector']), 0.0)
            self.inv_R = lambda t: np.array([[(1 - np.sin(theta*t)**2)/np.cos(theta*t), np.sin(theta*t)], [-np.sin(theta*t), np.cos(theta*t)]])
            self.c = lambda t: np.array([s*t for s in speed]).transpose()
        else:
            self.rigid_body_motion = False

    def _load_nonconformal_parameters(self) -> None:
        """
        Helper to load parameters needed for generating a nonconformal mesh and phase fields.
        """

        # Need to know the dimensionality of the simulation.
        self.dim = self.config.get_item(['DIM', 'mesh_dimension'], int)
        if self.dim not in [2, 3]:
            raise ValueError('The diffuse interface method is only implemented in 2D and 3D.')

        # Load the nonconformal mesh parameters.
        N_mesh = self.config.get_dict(['DIM', 'num_mesh_elements'], ngs.Parameter(0.0))
        scale = self.config.get_dict(['DIM', 'mesh_scale'], ngs.Parameter(0.0))
        offset = self.config.get_dict(['DIM', 'mesh_offset'], ngs.Parameter(0.0))

        # Check if phi should be created on a more refined mesh.
        N = self.config.get_dict(['DIM', 'num_phi_mesh_elements'], ngs.Parameter(0.0), quiet=True)
        if not N:
            # Just use the mesh spacing of the full mesh to create phi.
            N = N_mesh

        if self.dim == 2:
            self.N = [int(N['x']), int(N['y'])]
            self.N_mesh = [int(N_mesh['x']), int(N_mesh['y'])]
            self.scale = [scale['x'], scale['y']]
            self.offset = [offset['x'], offset['y']]
        elif self.dim == 3:
            self.N = [int(N['x']), int(N['y']), int(N['z'])]
            self.N_mesh = [int(N_mesh['x']), int(N_mesh['y']), int(N_mesh['z'])]
            self.scale = [scale['x'], scale['y'], scale['z']]
            self.offset = [offset['x'], offset['y'], offset['z']]

        # Load the interface parameters.
        self.lmbda = self.config.get_item(['DIM', 'interface_width_parameter'], float)

        # Load some parameters specific to how the interface approximation is created. These are highly optional.
        self.mnum = self.config.get_item(['DIM', 'mnum'], float, quiet=True)
        self.close = self.config.get_item(['DIM', 'close'], bool, quiet=True)

        # Dictates whether or not to invert phi.
        self.invert = self.config.get_item(['PHASE FIELDS', 'invert_phi'], bool, quiet=True)

    def _load_bc_parameters(self, config: ConfigParser, bc_config: ConfigParser, quiet: bool = False) -> None:
        """
        Helper function to load the BC parameters. Should only be called if self.multiple_bcs=True.

        Args:
            config: The main DIM config file.
            bc_config: The DIM BC config file (or the config file containing [VERTICES] and [CENTROIDS]).
            quiet: If True suppresses warnings about using default paraneter values.
        """
        self.vertices, _ = bc_config.get_one_level_dict('VERTICES', 0.0) # There should be no reason to ever re-parse vertices.

        try:
            self.centroid, _ = bc_config.get_one_level_dict('CENTROIDS', 0.0) # There should be no reason to ever re-parse centroid.
        except KeyError:
            self.centroid = {}

        self.lmbda_overlap = config.get_item(['DIM BOUNDARY CONDITIONS', 'overlap_interface_parameter'], float, quiet)
        if self.lmbda_overlap == -1:
            self.lmbda_overlap = False

        self.remainder = config.get_item(['DIM BOUNDARY CONDITIONS', 'remainder'], bool, quiet)

    def _generate_DIM_mesh(self) -> None:
        """
        Function to get the nonconformal mesh. This mesh is a simple rectangle or rectangular prism, is constructed of
        quadrilateral/hexahedral elements and has exterior boundaries "left", "right", "top", "bottom" (and
        "front" and "back" for 3D). See get_Netgen_nonconformal in mesh_helpers for more details.
        """

        self.ngmesh = mesh_helpers.get_Netgen_nonconformal(self.N_mesh, self.scale, self.offset, self.dim)
        self.mesh = ngs.Mesh(self.ngmesh)

        # Also create a refined mesh to calculate Grad(phi) and |Grad(phi)| on before projecting onto the coarse
        # simulation mesh.
        self.ngmesh_refined = mesh_helpers.get_Netgen_nonconformal(self.N, self.scale, self.offset, self.dim)
        self.mesh_refined = ngs.Mesh(self.ngmesh_refined)

    def _generate_phase_field(self):
        """
        Function to generate the phase field. This function produces a numpy array, which must later be converted into
        an NGSolve gridfunction.
        """

        if self.dim == 2:
            self.boundary_lst, bounds_lst = mesh_helpers.get_mesh_boundary_2d(self.stl_filename)
            # Check that the .stl file doesn't exceed the bounds of the domain. If it does, use the bounds of the .stl
            # instead, then trim the resulting phase field to fit the domain.
            self.tmp_N, self.tmp_scale, self.tmp_offset = mesh_helpers.get_new_bounds(bounds_lst, self.N, self.scale,
                                                                                      self.offset)
            binary_arr = interface.get_binary_2d(self.boundary_lst, self.tmp_N, self.tmp_scale, self.tmp_offset)

        elif self.dim == 3:
            self.boundary_lst, bounds_lst, z_bounds = mesh_helpers.get_stl_faces(self.stl_filename)
            # Check that the .stl file doesn't exceed the bounds of the domain. If it does, use the bounds of the .stl
            # instead, then trim the resulting phase field to fit the domain.
            self.tmp_N, self.tmp_scale, self.tmp_offset = mesh_helpers.get_new_bounds(bounds_lst, self.N, self.scale,
                                                                                      self.offset)
            binary_arr = interface.get_binary_3d(self.boundary_lst, self.tmp_N, self.tmp_scale, self.tmp_offset,
                                                      self.mnum, self.close)

        tmp_phi_arr = interface.get_phi(binary_arr, self.lmbda, self.tmp_N, self.tmp_scale, self.tmp_offset, self.dim)

        # Recombine tmp_phi_arr onto the array corresponding to the full nonconformal domain.
        phi_arr = mesh_helpers.crop_to_mesh_bounds(tmp_phi_arr, self.N, self.scale, self.offset, self.tmp_N,
                                                        self.tmp_scale, self.tmp_offset)

        # Invert phi if desired.
        if self.invert:
            phi_arr = 1.0 - phi_arr

        # Set zero areas to a small constant to prevent singularities.
        phi_arr = phi_arr * (1.0 - 1e-10) + 1e-10

        return phi_arr

    def _generate_BC_masks(self):
        """
        Function to generate the boundary condition phase field masks. This function produces numpy arrays, which must
        later be converted into NGSolve gridfunctions.
        """

        if self.dim == 2:
            tmp_mask_arr_dict = interface.split_nonconformal_subdomains_2d(self.boundary_lst, self.vertices,
                                                                            self.tmp_N, self.tmp_scale, self.tmp_offset,
                                                                            self.lmbda_overlap, self.remainder,
                                                                            self.centroid)
            # Recombine all masks onto the array corresponding to the full nonconformal domain.
            for marker, tmp_arr in tmp_mask_arr_dict.items():
                arr = mesh_helpers.crop_to_mesh_bounds(tmp_arr, self.N, self.scale, self.offset, self.tmp_N,
                                                       self.tmp_scale, self.tmp_offset)
                self.mask_arr_dict[marker] = arr

        elif self.dim == 3:
            tmp_mask_arr_dict = interface.split_nonconformal_subdomains_3d(self.boundary_lst, self.vertices,
                                                                            self.tmp_N, self.tmp_scale, self.tmp_offset,
                                                                            self.lmbda_overlap, self.remainder,
                                                                            self.centroid)
            # Recombine all masks onto the array corresponding to the full nonconformal domain.
            for marker, tmp_arr in tmp_mask_arr_dict.items():
                arr = mesh_helpers.crop_to_mesh_bounds(tmp_arr, self.N, self.scale, self.offset, self.tmp_N,
                                                       self.tmp_scale, self.tmp_offset)
                self.mask_arr_dict[marker] = arr

        return

    def get_DIM_gridfunctions(self, mesh: ngs.Mesh, interp_ord: int):
        """
        Function to get all of the phase fields and masks as gridfunctions. This is either done by loading them from
        files or by constructing the numpy arrays and then converting those into gridfunctions.

        Args:
            mesh: The mesh for the gridfunctions.
            interp_ord: The interpolant order for the finite element space for the gridfunctions.
        """

        if self.load_method == 'file':
            # The phase field and masks have already been created and saved as gridfunctions.
            fes = ngs.H1(mesh, order=interp_ord)

            phi_filename = self.config.get_item(['PHASE FIELDS', 'phi_filename'], str)
            self.phi_gfu = create_and_load_gridfunction_from_file(phi_filename, fes)

            if self.multiple_bcs:
                # There are BC masks that must be loaded.
                for marker, filename in self.vertices.items():
                    mask_gfu = create_and_load_gridfunction_from_file(filename, fes)
                    self.mask_gfu_dict[marker] = mask_gfu
            else:
                # One single mask that is just a grid function of ones.
                mask_gfu = ngs.GridFunction(fes)
                mask_gfu.Set(1.0)
                self.mask_gfu_dict['all'] = mask_gfu

            # Construct Grad(phi) and |Grad(phi)|.
            self.grad_phi_gfu = ngs.Grad(self.phi_gfu)
            self.mag_grad_phi_gfu = ngs.Norm(ngs.Grad(self.phi_gfu))

        else:
            # The phase field and masks must be generated as numpy arrays and then converted into gridfunctions.
            # The phase field array has already been generated because it is needed to generate the mesh.
            # Construct the phase field, Grad(phi), and |Grad(phi)|.
            if self.N == self.N_mesh:
                # phi was generated on the simulation mesh, so just load phi into a gridfunction and compute Grad(phi)
                # and |Grad(phi)|.
                self.phi_gfu = numpy_to_NGSolve(self.mesh, interp_ord, self.phi_arr, self.scale, self.offset, self.dim)
                self.grad_phi_gfu = ngs.Grad(self.phi_gfu)
                self.mag_grad_phi_gfu = ngs.Norm(ngs.Grad(self.phi_gfu))
            else:
                # phi was generated on a refined mesh, so load it into a refined mesh gridfunction, compute Grad(phi)
                # and |Grad(phi)|, then project all three into gridfunctions defined on the simulation mesh.
                phi_gfu_tmp = numpy_to_NGSolve(self.mesh_refined, interp_ord, self.phi_arr, self.scale, self.offset, self.dim)
                grad_phi_gfu_tmp = ngs.Grad(phi_gfu_tmp)
                mag_grad_phi_gfu_tmp = ngs.Norm(ngs.Grad(phi_gfu_tmp))

                # Now project onto the coarse simulation mesh.
                fes = ngs.H1(mesh, order=interp_ord)
                vec_fes = ngs.VectorH1(mesh, order=interp_ord)
                self.phi_gfu = ngs.GridFunction(fes)
                self.grad_phi_gfu = ngs.GridFunction(vec_fes)
                self.mag_grad_phi_gfu = ngs.GridFunction(fes)

                self.phi_gfu.Set(phi_gfu_tmp)
                self.grad_phi_gfu.Set(grad_phi_gfu_tmp)
                self.mag_grad_phi_gfu.Set(mag_grad_phi_gfu_tmp)

            if self.multiple_bcs:
                # There are multiple BC masks that must be generated and loaded.
                for marker, mask_arr in self.mask_arr_dict.items():
                    mask_gfu = numpy_to_NGSolve(mesh, interp_ord, mask_arr, self.scale, self.offset, self.dim)
                    self.mask_gfu_dict[marker] = mask_gfu
            else:
                # One single mask that is just a grid function of ones.
                mask_arr = np.ones(tuple([int(n + 1) for n in self.N]))
                mask_gfu = numpy_to_NGSolve(mesh, interp_ord, mask_arr, self.scale, self.offset, self.dim)
                self.mask_gfu_dict['all'] = mask_gfu

            # Save the gridfunctions if desired.
            save_to_file = self.config.get_item(['PHASE FIELDS', 'save_to_file'], bool)
            if save_to_file:
                self.phi_gfu.Save(self.DIM_dir + '/phi.sol')
                self.ngmesh.Save(self.DIM_dir + '/mesh.vol')

                for marker, gfu in self.mask_gfu_dict.items():
                    # Save the BC masks inside the DIM bc_dir.
                    gfu.Save(self.DIM_dir + '/bc_dir/{}_mask.sol'.format(marker))

