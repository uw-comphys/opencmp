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

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Union, cast

import ngsolve as ngs
from ngsolve.comp import ProxyFunction, FESpace, DifferentialSymbol
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm, Preconditioner, CoefficientFunction
from pyngcore import BitArray

from ..helpers import merge_bc_dict
from ..config_functions import ConfigParser, BCFunctions, ICFunctions, ModelFunctions, RefSolFunctions
from ..config_functions.load_config import parse_str
from ..diffuse_interface import DIM
from ..helpers.io import load_mesh
from ..helpers.error import norm, mean

"""
Module for the base model class.
"""


class Model(ABC):
    """
    Base class which other models will subclass from.
    """

    def __init__(self, config: ConfigParser, t_param: List[Parameter]) -> None:
        """
        Initializer for the model classes

        Args:
            config: A configparser object loaded with config file.
            t_param: List of parameters representing the current and previous timestep times.
        """
        # The name of the model (helper variable). It is the name of the class.
        self.name = self.__class__.__name__.lower()

        logging.info('Initializing model: ' + self.name)

        # Remove the trailing "dim" from the model name if the Diffuse Interface Method version is used
        # Lets us continue using "Poisson" or "INS" in config files without adding the DIM
        if "dim" == self.name[-3:]:
            self.name = self.name[:-3]

        # Set config file.
        self.config = config

        # Set time stepping parameter.
        self.t_param = t_param

        # Get the run directory.
        self.run_dir = self.config.get_item(['OTHER', 'run_dir'], str)

        logging.info('Initializing with run directory: ' + self.run_dir)

        # Dictionary to map variable name to index inside fes/gridfunctions, and for labeling variables in vtk output.
        # These are currently the same, but will be changed in other functions. E.g. MCINS._add_multiple_components()
        self.model_components    = self._define_model_components()
        self.model_components_ic = self._define_model_components()

        # Dictionary to specify if a particular component should be used for error calculations
        self.model_local_error_components = self._define_model_local_error_components()

        # List of dictionaries to specify if a particular component should have a time derivative.
        # Each dictionary in the list corresponds to a different weak form for the model.
        self.time_derivative_components = self._define_time_derivative_components()

        # The number of weak forms used by the model. E.g. the drift-flux model has two separate weak forms
        # due to its discretization scheme. conservation of mass and momentum as de-coupled and solved sequentially
        # thus each needs an independent weak form.
        self.num_weak_forms = self._define_num_weak_forms()

        # Ensure that local error calculations are specified for each component
        assert len(self.model_components)\
               == len(self.model_local_error_components)
        # Ensure that each weak form has derivatives associated with it
        assert self.num_weak_forms == len(self.time_derivative_components)
        # Ensure that every variable is specified for the time derivative terms for each weak form
        for form in self.time_derivative_components:
            assert len(self.model_components) == len(form)

        # For MCINS, whether or not the velocity is fixed in time at the initial condition, and thus does not have to be solved
        self.fixed_velocity = self.config.get_item(['OTHER', 'velocity_fixed'], bool)

        logging.info('Initializing using fixed velocity for convective fluxes.')

        # Run any model-specific work now that some things have been initialized.
        self._pre_init()

        # Create a list of dictionaries to hold the values of any model variables or parameters that are variables in
        # model function definitions (ex: if source = u * diffusion_coefficients_a then this list of dictionaries holds
        # the values of u and diffusion_coefficients_a at each time step in the time discretization scheme).
        self.update_variables = [self.model_components.copy() for _ in self.t_param]
        self._add_multiple_parameters(config)

        # Invert the model components dictionary, so that you can loop up component name by index
        self.model_components_inverted = dict(zip(self.model_components.values(), self.model_components.keys()))

        # Check if the diffuse interface method is being used and if yes initialize a DIM class object.
        self.DIM = self.config.get_item(['DIM', 'diffuse_interface_method'], bool, quiet=True)
        if self.DIM:
            self.DIM_dir = self.config.get_item(['DIM', 'dim_dir'], str)
            self.DIM_solver = DIM(self.DIM_dir, self.run_dir, self.t_param)

        # Load the mesh. If the diffuse interface method is being used the mesh will be constructed/loaded by the DIM
        # solver.
        self.load_mesh_fes(mesh=True, fes=False)

        # Load the finite element space parameters.
        self.element = self.config.get_dict(['FINITE ELEMENT SPACE', 'elements'], self.run_dir, None, all_str=True)
        self.interp_ord = self.config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)

        # Ensure that each component has an associated element
        assert len(self.model_components) == len(self.element) - (2 if self.fixed_velocity else 0)

        # Check if DG should be used.
        self.DG = self.config.get_item(['DG', 'DG'], bool)

        if self.DG:
            # Load the DG parameters.
            self.ipc = self.config.get_item(['DG', 'interior_penalty_coefficient'], float)
            logging.info('Using DIScontinuous Galerkin method')
        else:
            # Need to define this parameter, but won't end up using it so suppress the default value warning.
            self.ipc = self.config.get_item(['DG', 'interior_penalty_coefficient'], float, quiet=True)
            logging.info('Using continuous Galerkin method')

        self.no_constrained_dofs = self.config.get_item(['FINITE ELEMENT SPACE', 'no_constrained_dofs'],
                                                        bool, quiet=True)

        if 'HDiv' in self.element.values() or 'RT' in self.element.values():
            # HDiv elements can only strongly apply u.n Dirichlet boundary conditions. In order to apply other
            # Dirichlet boundary conditions (ex: on the full vector or the tangential vector) the boundary
            # conditions must be weakly imposed. This is done in DG by penalization terms but requires solving
            # over all DOFs, not just the free DOFs. This will not work in CG, hence the warning to the user.
            if self.no_constrained_dofs:
                if not self.DG:
                    logging.warning('It is strongly recommended to use DG with HDiv if tangential Dirichlet boundary conditions need to be applied.')

        # TODO: we should probably rename self.nu to something else so it doesnt conflict with kinematic viscosity
        # Construct nu as ipc*interp_ord^2.
        self.nu = self.ipc * self.interp_ord ** 2

        # Load the solver parameters.
        self.linear_solver = self.config.get_item(['SOLVER', 'linear_solver'], str)
        if self.linear_solver == 'default':
            # Default to a direct solve.
            self.linear_solver = 'direct'

        if self.linear_solver == 'direct':
            logging.warning("The direct linear solver does not respect the num_threads parameter you may have set. \\This is an NGSolve issue")

        # Load the preconditioner types
        self.preconditioners = self.config.get_list(['SOLVER', 'preconditioner'], str)
        # Ensure that enough have been specified
        assert len(self.preconditioners) == self.num_weak_forms

        # Perform checks on preconditioner types
        for i in range(len(self.preconditioners)):
            if self.preconditioners[i] == 'default':
                # 'local' seems to be the default preconditioner in the NGSolve examples.
                self.preconditioners[i] = 'local'
            if self.preconditioners[i] == 'None':
                # Parse the string value.
                self.preconditioners[i] = None
                if self.linear_solver in ['CG', 'MinRes']:
                    # CG and MinRes require a preconditioner or they fail with an AssertionError.
                    logging.error('Preconditioner cannot be None if using CG or MinRes solvers.')
                    raise ValueError('Preconditioner cannot be None if using CG or MinRes solvers.')

        self.linear_tolerance = self.config.get_item(['SOLVER', 'linear_tolerance'], float, quiet=True)
        self.linear_max_iterations = self.config.get_item(['SOLVER', 'linear_max_iterations'], int, quiet=True)

        # assume model is linear by default
        self.nonlinear = False

        self.verbose = self.config.get_item(['OTHER', 'messaging_level'], int, quiet=True) > 0

        # Initialize the BC functions class
        # Note: Need dirichlet_names before constructing fes, but need to construct fes before gridfunctions can be
        #       loaded for ex: BCs, ICs, reference solutions. So, get the BC dict before constructing fes,
        #       but then actually load the BCs etc after constructing fes.
        self.bc_functions = BCFunctions(self.run_dir + '/bc_dir/bc_config', self.run_dir, self.mesh, self._define_bc_types(),
                                        self.t_param, self.update_variables)

        # 1/2: Create BCs.
        # Needs to be done in two steps since initializing the BC gridfunctions requires a FES, which in turn requires
        # info about the location of dirichlet BCs.
        # Here we define the BCs and load as much of them as possible
        self.BC, self.dirichlet_names = self.bc_functions.set_boundary_conditions(self._define_bc_types())

        # Create the finite element space.
        # This needs to be done before the initial conditions are loaded.
        self.fes = self._construct_fes()
        self.fes_ic = self._construct_ic_fes()

        self._test, self._trial = self._create_test_and_trial_functions()

        # Create the rest of the ConfigFunction classes
        self.ic_functions      = ICFunctions(self.run_dir + '/ic_dir/ic_config', self.run_dir, self.mesh, self.t_param,
                                             self.update_variables)
        self.model_functions   = ModelFunctions(self.run_dir + '/model_dir/model_config', self.run_dir, self.mesh,
                                                self.t_param, self.update_variables)
        self.ref_sol_functions = RefSolFunctions(self.run_dir + '/ref_sol_dir/ref_sol_config', self.run_dir, self.mesh,
                                                 self.t_param, self.update_variables)

        # Load initial condition.
        self.IC = self.construct_gfu_ic()
        self.ic_functions.set_initial_conditions(self.IC, self.mesh, self.name, self.model_components_ic)

        # 2/2: Create BCs
        # Now that we have the FES, we can load in the bc gridfunctions
        # Load any boundary conditions saved as gridfunctions.
        self.BC = self.bc_functions.load_bc_gridfunctions(self.BC, self.fes, self.model_components)
        # TODO: Give good description of this variable here
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu_ic(),
                                                                       self.model_components)

        # Load the model functions and model parameters.
        self.model_functions.set_model_functions(self.fes, self.model_components)
        self._set_model_parameters()

        # Load exact solution.
        self.ref_sol = self.ref_sol_functions.set_ref_solution(self.fes, self.model_components)

        # Names to call variables in vtk output
        self.save_names: List[str] = list(self.model_components.keys())

        # If the diffuse interface method is being used generate the phase field and BC masks and construct the
        # dictionary of BC values. Note that these are only the BCs to be applied to the nonconformal boundary. There
        # may also be BCs applied to the conformal boundary, which are contained in self.BC as usual.
        if self.DIM:
            self.DIM_solver.get_DIM_gridfunctions(self.mesh, self.interp_ord)
            # DIM_dirichlet_names should never be used and a DIM g_D is not needed.
            self.DIM_bc_functions = BCFunctions(self.DIM_dir + '/bc_dir/dim_bc_config', self.run_dir, self.mesh,
                                                self._define_bc_types(),
                                                self.t_param, self.update_variables)
            self.DIM_BC, DIM_dirichlet_names = self.DIM_bc_functions.set_boundary_conditions(self._define_bc_types())
            self.DIM_BC = self.DIM_bc_functions.load_bc_gridfunctions(self.DIM_BC, self.fes, self.model_components)

        # By default assume that the model does not need to be linearized.
        self.linearize = None

        # Do any model-specific work now that the model has been initialized
        self._post_init()

    def _add_multiple_parameters(self, config: ConfigParser) -> None:
        """
        Function to add multiple time-varying parameters (whose values are not known in closed form) to the model from
        the config file.

        Args:
            config: The ConfigParser
        """

        # Name of time-varying parameters.
        self.model_parameters = config.get_list(['OTHER', 'parameter_names'], str, quiet=True)

        # Initialize all time values of each time-varying parameter to zero.
        for param in self.model_parameters:
            # Should never be using the same name for a model component and parameter.
            assert param not in self.update_variables[0].keys()

            for time_dict in self.update_variables:
                time_dict[param] = 0

    def _create_test_and_trial_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        """
        Function to create the trial and test (weighting) function(s) for the model.

        Returns:
            Returns two lists, one for the trial and test (weighting) function(s).
        """
        # Get test and trial functions
        trial = self.fes.TrialFunction()
        test  = self.fes.TestFunction()

        # For single variable models, trial and test will be a single value, we want a list
        if type(trial) is not list:
            trial = [trial]
            test = [test]

        return test, trial

    def _construct_linearization_terms(self) -> Optional[List[GridFunction]]:
        """
        Function to construct the list of linearization terms.

        If a specific model does not need linearization terms, return None.

        Returns:
            None or a list of grid functions used to linearize a non-linear model.
        """

        return None

    def _ds(self, marker: str) -> DifferentialSymbol:
        """
        Function to get the appropriate (DG vs non-DG) version of ds for the given marker.

        Args:
            marker: String representing the mesh marker to get ds on.

        Returns:
            The appropriate ds.
        """
        return ngs.ds(skeleton=self.DG, definedon=self.mesh.Boundaries(marker))

    def apply_dirichlet_bcs_to(self, gfu: GridFunction, time_step: int = 0) -> None:
        """
        Function to set the Dirichlet boundary conditions within the solution GridFunction.

        Args:
            gfu: The GridFunction to add the Dirichlet boundary condition values to.
            time_step: Specifies which time step's boundary condition values to use. This would usually be 0 (the
                t^n+1 values) except in the case of adaptive_three_step and Runge Kutta schemes.
        """
        # NOTE: DO NOT change from definedon=self.mesh.Boundaries(marker) to definedon=marker.
        if len(self.g_D) > 0:
            if len(gfu.components) == 0:  # Single trial functions
                # TODO: IDE is complaining that we don't specify parameter VOL_OR_BND for .Set()
                gfu.Set(self.g_D['u'][time_step], definedon=self.mesh.Boundaries(self.dirichlet_names['u']))
            else:  # Multiple trial functions.
                for component_name in self.g_D.keys():
                    i = self.model_components[component_name]
                    # Apply Dirichlet or pinned BCs, but only to non-L2 space
                    if 'L2' not in self.fes.components[i].name:
                        gfu.components[i].Set(self.g_D[component_name][time_step],
                                              definedon=self.mesh.Boundaries(self.dirichlet_names[component_name]))

    def construct_gfu(self) -> GridFunction:
        """
        Function to construct a solution GridFunction.

        Returns:
            A GridFunction initialized on the finite element space of the model.
        """
        gfu = ngs.GridFunction(self.fes)

        return gfu

    def construct_gfu_ic(self) -> GridFunction:
        """
        Function to construct a solution GridFunction.

        Returns:
            A GridFunction initialized on the finite element space of the model.
        """
        gfu = ngs.GridFunction(self.fes_ic)

        return gfu

    def construct_preconditioners(self, a_assembled: List[BilinearForm]) -> List[Preconditioner]:
        """
        Function to construct the preconditioners needed by the model.

        Args:
            a_assembled: A list of the assembled bilinear form.

        Returns:
            A list of the constructed preconditioners.
        """
        contructed_preconditioners: List[Preconditioner] = []

        for i in range(len(self.preconditioners)):
            if self.preconditioners[i] is None:
                contructed_preconditioners.append(None)
            else:
                contructed_preconditioners.append(ngs.Preconditioner(a_assembled[i], self.preconditioners[i]))

        return contructed_preconditioners

    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        """
        Function return the trial and test (weighting) function(s) for the model.

        Returns:
            Returns two lists, one for the trial and one for the test (weighting) function(s).
        """
        return self._trial, self._test

    def load_mesh_fes(self, mesh: bool = True, fes: bool = True):
        """
        Function to load the model's mesh.

        Args:
            mesh: If True, reload the original mesh from file.
            fes: If True, reconstruct the finite element space.
        """
        if mesh:
            # Load/reload the mesh.
            if self.DIM and (self.DIM_solver.load_method == 'generate' or self.DIM_solver.load_method == 'combine'):
                try:
                    # TODO: Elizabeth this is a hack and needs to be fixed or it will blow up in your face.
                    self.mesh = load_mesh(self.config)
                except:
                    self.mesh = self.DIM_solver.mesh
            else:
                self.mesh = load_mesh(self.config)

        if fes:
            # Load/reload the finite element space.
            self.fes = self._construct_fes()

    # TODO: Move to time_integration_schemes.py
    def time_derivative_terms(self, gfu_lst: List[List[GridFunction]], scheme: str, step: int = 1) \
            -> Tuple[List[CoefficientFunction], List[CoefficientFunction]]:
        """
        Function to produce the time derivative terms for the linear and bilinear forms.

        Args:
            gfu_lst: List of the solutions of previous time steps in reverse chronological order.
            scheme: The name of the time integration scheme being used.
            step: Which intermediate step the time derivatives are needed for. This is specific to Runge Kutta schemes.

        Returns:
            Tuple[CoefficientFunction, CoefficientFunction]:
                - a: The time derivative terms for the bilinear forms.
                - L: The time derivative terms for the linear forms.
        """
        U, V = self.get_trial_and_test_functions()

        # List of indices to ignore for the purpose of calculating time derivative
        ignore_indices: List[List[int]] = [[] for _ in range(self.num_weak_forms)]

        # Populate the list
        # NOTE: We assume that if a model component has it's error calculated
        #       it should also be added to the time derivative
        if len(U) > 1:
            for form in range(len(self.time_derivative_components)):
                ignore_indices.append([])
                for val in self.time_derivative_components[form]:
                    if not self.time_derivative_components[form][val]:
                        index = self.model_components[val]
                        if index is None:
                            raise ValueError("Variable \"{}\" was not expected to have an index of \"None\""
                                             "since it's not the lone variable for the model.".format(val))
                        else:
                            ignore_indices[form].append(cast(int, index))

        # Create the terms for the linear and bilinear form into which to add the time derivative
        a: List[CoefficientFunction] = []
        L: List[CoefficientFunction] = []

        # Loop over each weak form
        for j in range(self.num_weak_forms):
            a_tmp = ngs.CoefficientFunction(0.0)
            L_tmp = ngs.CoefficientFunction(0.0)

            # Loop over each trial function
            for i in range(len(U)):
                # TODO: WHY DOES THIS GET OUT OF BOUNDS
                if i not in ignore_indices[j]:
                    # TODO: Figure out a better way to do this so that the string is stored elsewhere. Maybe a named tuple.
                    if scheme == 'explicit euler':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[1][i] * V[i]
                    elif scheme == 'implicit euler':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[1][i] * V[i]
                    elif scheme == 'crank nicolson':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[1][i] * V[i]
                    elif scheme == 'adaptive imex pred':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[1][i] * V[i]
                    elif scheme == 'CNLF':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[2][i] * V[i]
                    elif scheme == 'SBDF':
                        a_tmp += (11 / 6) * U[i] * V[i]
                        L_tmp += (3 * gfu_lst[1][i] - 1.5 * gfu_lst[2][i] + 1 / 3 * gfu_lst[3][i]) * V[i]
                    elif scheme == 'RK 222':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[step][i] * V[i]
                    elif scheme == 'RK 232':
                        a_tmp += U[i] * V[i]
                        L_tmp += gfu_lst[step][i] * V[i]
                    else:
                        raise ValueError("Scheme \"{}\" is not implemented".format(scheme))

            a.append(a_tmp)
            L.append(L_tmp)

        return a, L

    def update_bcs(self, bc_dict_patch: Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]]) -> None:
        """
        Function to update BCs to arbitrary values.

        This function is used to implement controllers. It lets them manipulate the manipulated variable.

        Args:
            bc_dict_patch: Dictionary containing new values for the BCs being updated.
        """
        # Merge the bc dictionary
        self.BC = merge_bc_dict(self.BC, bc_dict_patch)

        # Reload everything as grid functions
        self.BC = self.bc_functions.load_bc_gridfunctions(self.BC, self.fes, self.model_components)
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu(),
                                                                       self.model_components)

    def update_model_variables(self, updated_gfu: Union[GridFunction, List[ProxyFunction]], ic_update: bool = False,
                               ref_sol_update: bool = False, time_step: Optional[int] = None) -> None:
        """
        Function to update the values of the model variables and then re-parse any expressions that contain those model
        variables in ex: the model functions, the model boundary conditions...

        Args:
            updated_gfu: Gridfunction containing updated values for all model variables.
            ic_update: If True update the initial conditions.
            ref_sol_update: If True update the reference solutions.
            time_step: If specified, only update the model component values for one particular time step.
        """

        # Update the values of the model variables.
        if time_step is not None:
            # Only updating the values for one specific step.
            if isinstance(updated_gfu, list):
                # List of trial functions.
                for key in self.model_components.keys():
                    component = self.model_components[key]

                    if component is None:
                        # One single model variable.
                        self.update_variables[time_step][key] = updated_gfu[0]

                    else:
                        # Multiple model variables.
                        self.update_variables[time_step][key] = updated_gfu[component]

            elif isinstance(updated_gfu, GridFunction):
                # Gridfunction with updated model component values.
                for key in self.model_components.keys():
                    component = self.model_components[key]

                    if component is None:
                        # Gridfunction over the full finite element space.
                        self.update_variables[time_step][key] = updated_gfu
                    else:
                        # Component of the finite element space.
                        self.update_variables[time_step][key] = updated_gfu.components[component]
        else:
            # Confirm that a gridfunction was given, not a list of trial functions. It only makes sense to insert a
            # list of trial functions into the values for the current to-be-solved-for time step, not to update all the
            # time step values.
            assert isinstance(updated_gfu, GridFunction)

            # If len(self.update_variables) = 1 that means a stationary solve is occurring and it makes no sense
            # to update the model variable values with information from previous time steps.
            assert len(self.update_variables) > 1

            # First shift values at previous time steps to reflect that a new time step has started.
            for i in range(1, len(self.update_variables)-1):
                self.update_variables[-i] = self.update_variables[-(i + 1)].copy()

            # Then get the values for the just solved time step.
            for key in self.model_components.keys():
                component = self.model_components[key]

                if component is None:
                    # Gridfunction over the full finite element space.
                    self.update_variables[1][key] = updated_gfu
                else:
                    # Component of the finite element space.
                    self.update_variables[1][key] = updated_gfu.components[component]

        # Update the values of any time-varying model parameters used in other expressions (model functions, BCs, etc).
        for key in self.model_parameters:
            # Weirdness needed to account for model_parameters_dict being a two-level dictionary. Ex: if
            # model_parameters_values contains a parameter called diffusion_coefficients_a then the value of that
            # parameter is contained in model_parameters_dict[diffusion_coefficients][a].
            parameter, var = self.model_functions.model_parameters_names[key]
            if var in self.model_functions.model_parameters_re_parse_dict[parameter].keys():
                # Update the value of the parameter either by calling some imported Python function or just by
                # re-parsing the string expression.
                re_parse_expression = self.model_functions.model_parameters_re_parse_dict[parameter][var]
                if callable(re_parse_expression):
                    val_lst = [re_parse_expression(self.t_param, self.update_variables, self.mesh, i) for i in range(len(self.t_param))]

                else:
                    val_lst, _ = parse_str(self.model_functions.model_parameters_re_parse_dict[parameter][var],
                                           self.run_dir, self.t_param, self.update_variables, mesh=self.mesh)

            else:
                # Just grab the value of the parameter from model_functions.model_parameters_dict. Note that this needs
                # to be done or the value of the parameter will remain the initialized value (0).
                val_lst = self.model_functions.model_parameters_dict[parameter][var]

            # Re-parsing gives a list of values corresponding to each time step in the time discretization scheme.
            # Assign these values to the correct dictionary in model_parameter_values.
            for i in range(len(self.update_variables)):
                self.update_variables[i][key] = val_lst[i]

        # Re-parse any expressions as necessary with the new model component and model parameter values.
        #
        # Start with the boundary conditions.
        self.bc_functions.update_boundary_conditions(self.t_param, self.update_variables, self.mesh)
        self.BC, dirichlet_names = self.bc_functions.set_boundary_conditions(self.BC)
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu(),
                                                                       self.model_components)

        # Also update the DIM boundary conditions if they exist.
        if self.DIM:
            self.DIM_bc_functions.update_boundary_conditions(self.t_param, self.update_variables, self.mesh)
            self.DIM_BC, DIM_dirichlet_names = self.DIM_bc_functions.set_boundary_conditions(self.DIM_BC)

        # Update the model parameters and functions.
        self.model_functions.update_model_functions(self.t_param, self.update_variables, self.mesh)
        self._set_model_parameters()

        if ic_update:
            # Update the initial conditions.
            self.ic_functions.update_initial_conditions(self.t_param, self.update_variables, self.mesh)
            self.IC = self.construct_gfu_ic()
            self.ic_functions.set_initial_conditions(self.IC, self.mesh, self.name, self.model_components)

        if ref_sol_update:
            # Update the reference solutions.
            self.ref_sol_functions.update_ref_solutions(self.t_param, self.update_variables, self.mesh)
            self.ref_sol = self.ref_sol_functions.set_ref_solution(self.fes, self.model_components)

    def update_timestep(self, gfu: GridFunction, gfu_0: GridFunction) -> None:
        """
        Function to update the previous time-step gridfunction with result of the current time-step.

        Args:
            gfu: The gridfunction containing the result of the current time-step.
            gfu_0: The gridfunction containing the result of the previous time-step.
        """

        gfu_0.vec.data = gfu.vec

    @staticmethod
    @abstractmethod
    def allows_explicit_schemes() -> bool:
        """
        Function to specify whether a given model works with explicit time integration schemes.

        Returns:
            True if the model can be used with fully explicit time integration schemes, else False.
        """

    @abstractmethod
    def _construct_fes(self) -> FESpace:
        """
        Function to construct the finite element space for the model.

        Returns:
            The finite element space.
        """

    def _construct_ic_fes(self) -> FESpace:
        """
        Function to construct the finite element space for the model's initial condition.
        In more circumstances this is identical to self._construct_fes.
        The exception is the MCINS model when the velocity and pressure do not wish to be solved in time.
        In that case, this method will return a mixed function space which contains a function space for the velocity in pressure
        while the result from self._construct_fes will NOT have a function space for velocity and pressure.

        Returns:
            The finite element space onto which to load the initial condition
        """
        return self._construct_fes()

    @abstractmethod
    def _define_bc_types(self) -> List[str]:
        """
        Function to specify the types of boundary conditions (BCs) available for this model.

        Returns:
            List containing the names, as str, of all available/allowable BC types.
        """

    @abstractmethod
    def _define_model_components(self) -> Dict[str, Optional[int]]:
        """
        Function to specify the mapping, as a dictionary, between variable names and their position within the finite
        element space, and all other locations which derive their indexing order from the finite element space.

        For a single variable model, e.g. Poisson, the index value MUST be None. This is due to how NGSolve defines
        gridfunctions for single vs mixed finite element spaces.

        Returns:
            Returns a dictionary whose keys are the names of the trial functions and values are the index of a
            given trial function within the finite element space.
        """

    @abstractmethod
    def _define_model_local_error_components(self) -> Dict[str, bool]:
        """
        Function to specify which model components (trial functions) should be included in the local error calculation.

        Returns:
            Returns a dictionary whose keys are the names of the trial functions and values is a bool which
            indicates if the local error should be calculated for that trial function.
        """

    @abstractmethod
    def _define_num_weak_forms(self) -> int:
        """
        Function to specify the number of seperate weak forms this model has.

        This is mostly used if a linearization scheme is used in order to de-couple multiple equations.
        E.g. Solving INS by solving conservation of mass first, using the previous iteration/time-step's conservation
        of momentum values, and then use that result to solve for conservation of momentum.

        Returns:
            Int representing the number of weak forms.
        """

    @abstractmethod
    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        """
        Function to specify which trial functions have a time derivative associated with them.
        Additionally, it specifies in which of the model's (potentially) multiple weak forms the time derivative should
        be present.

        The weak form must be derived so that the time derivative has a coefficient of 1, i.e. not multiplied by
        anything.

        Returns:
            A list of dictionaries, one for each weak form of the model (in order). The keys for each are the name
            of the trial function and value is a bool indicating if this trial function has a time derivative in the
            weak form represented by the index of the dictionary within the list.
        """

    @abstractmethod
    def _post_init(self) -> None:
        """
        Function to do extra model-specific work after the ENTIRE Model.__init__() has run.

        This approach is used, instead of letting the user override __init__() and then calling super().__init__() from
        their custom model, so that we can make it explicit which parts are inherited from which parent.
        """

    @abstractmethod
    def _pre_init(self) -> None:
        """
        Function to do extra model-specific work after _define_model_components, _define_model_local_error_components,
        _define_time_derivative_components, _define_num_weak_forms, and _define_bc_types have run.
        """

    @abstractmethod
    def _set_model_parameters(self) -> None:
        """
        Function to initialize model parameters and functions from the configfile.
        """

    @abstractmethod
    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                    dt: Parameter, time_step: int) -> List[BilinearForm]:
        """
        Function to construct a portion of the bilinear form for model variables WITHOUT time derivatives.

        A given model with multiple model variables may not include time derivatives of all of its model variables.
        It is necessary to split the bilinear form into two portions as some time discretization schemes use different
        coefficients for terms with and without time derivatives.

        Args:
            U: A list of trial functions for the model's finite element space.
            V: A list of testing (weighting) functions for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            List of the described portion of the bilinear form for the model. The list will contain multiple bilinear
            forms if the model involves iterating between different versions of a linearized weak form.
        """

    @abstractmethod
    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        """
        Function to construct a portion of the bilinear form for model variables WITH time derivatives (basically any
        bilinear form terms not already in construct_bilinear_time_coefficient).

        A given model with multiple model variables may not include time derivatives of all of its model variables.
        It is necessary to split the bilinear form into two portions as some time discretization schemes use different
        coefficients for terms with and without time derivatives.

        Args:
            U: A list of trial functions for the model's finite element space.
            V: A list of testing (weighting) functions for the model's finite element space.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            List of the described portion of the bilinear form for the model. The list will contain multiple bilinear
            forms if the model involves iterating between different versions of a linearized weak form.
        """

    @abstractmethod
    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:
        """
        Function to construct the linear form containing any terms that arise solely from an IMEX scheme.

        This function should only include the terms specific to the IMEX scheme. E. g. for INS only the explicit form of
        the convection term would be included in this function. The other linear form terms like the body force and
        boundary condition terms (which exist regardless of the linearization scheme) would be handled by
        construct_linear.

        Args:
            V: A list of test (weighting) functions for the model's finite element space.
            gfu_0: List of gridfunction components from previous time steps.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            List of the explicit IMEX terms to add to the linear form for the model. The list will contain multiple
            items if the model involves iterating between different versions of a linearized weak form.
        """

    @abstractmethod
    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:
        """
        Function to construct the linear form.

        DO NOT include any terms that arise from and an IMEX scheme, see construct_imex_explicit for those terms.

        Args:
            V: A list of test (weighting) functions for the model's finite element space.
            gfu_0: List of gridfunction components from previous time steps.
            dt: Time step parameter (in the case of a stationary solve is just one).
            time_step: What time step values to use for ex: boundary conditions. The value corresponds to the index of
                the time step in t_param and dt_param.

        Returns:
            List of the linear form for the model. The list will contain multiple linear forms if the model involves
            iterating between different versions of a linearized weak form.
        """

    @abstractmethod
    def solve_single_step(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                         precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:
        """
        Function to solve the model.

        This function produces the model solution after one complete time step (or just the steady state solution for a
        stationary solve). It includes an inner iterations needed for ex: Picard iterations with Oseen linearization or
        iteration between weak forms for a model with multiple weak forms.

        Args:
            a_lst: A list of the bilinear forms.
            L_lst: A list of the linear forms.
            precond_lst: A list of preconditioners to use.
            gfu: The gridfunction to store the solution from the current iteration.
            time_step: What time step values to use if _apply_dirichlet_bcs_to must be called.
        """

    def linear_solve(self, a_assembled: BilinearForm, L_assembled: LinearForm, precond: Preconditioner,
                                 gfu: GridFunction) -> None:
        """
        Function to solve of the linear system associated with the model.

        This function performs a linear solve of the linear system associated with the model.
        Args:
            a_assembled: The assembled bilinear form.
            L_assembled: The assembled linear form.
            precond: The preconditioner.
            gfu: The gridfunction holding information about any Dirichlet BCs which will be updated to hold the
                solution.
        """

        if precond is None:
            # If there's no preconditioner, then the freedofs argument must be provided for the solver.
            # Confirm that the user hasn't specified that all dofs should be free dofs.
            if self.no_constrained_dofs:
                raise ValueError('Must constrain Dirichlet DOFs if not providing a preconditioner.')

        freedofs: Optional[BitArray] = self.fes.FreeDofs()

        if self.linear_solver == 'direct':
            # prefer PARDISO if available, else use UMFPACK, note that pip version of NGSolve does
            # does not seem to populate the ngsolve.config.USE_PARDISO etc. variables correctly
            if ngs.config.USE_PARDISO or ngs.config.USE_MKL:
                inverse_solver = "pardiso"
            elif ngs.config.USE_UMFPACK:
                inverse_solver = "umfpack"
            else:
                raise NameError("NGSolve compiled without PARDISO or UMFPACK support.")

            inv = a_assembled.mat.Inverse(freedofs=freedofs, inverse=inverse_solver)

            r = L_assembled.vec.CreateVector()
            r.data = L_assembled.vec - a_assembled.mat * gfu.vec
            gfu.vec.data += inv * r

        elif self.linear_solver == 'CG':
            solver = ngs.solvers.CG(mat=a_assembled.mat, rhs=L_assembled.vec, pre=precond, sol=gfu.vec,
                           tol=self.linear_tolerance, maxsteps=self.linear_max_iterations, printrates=self.verbose,
                           initialize=False)

        elif self.linear_solver == 'MinRes':
            ngs.solvers.MinRes(mat=a_assembled.mat, rhs=L_assembled.vec, pre=precond, sol=gfu.vec,
                               tol=self.linear_tolerance, maxsteps=self.linear_max_iterations,
                               printrates=self.verbose)

        elif self.linear_solver == 'GMRes':
            ngs.solvers.GMRes(A=a_assembled.mat, b=L_assembled.vec, pre=precond, freedofs=freedofs,
                              x=gfu.vec, tol=self.linear_tolerance, maxsteps=self.linear_max_iterations,
                              printrates=self.verbose)

        elif self.linear_solver == 'Richardson':
            # TODO: User should be able to set a damping factor.
            ret = ngs.solvers.PreconditionedRichardson(a=a_assembled, rhs=L_assembled.vec, pre=precond,
                                                       freedofs=freedofs, tol=self.linear_tolerance,
                                                       maxit=self.linear_max_iterations, printing=self.verbose)
            gfu.vec.data = ret
        else:
            logging.error('No linear solver specified.')
            raise ValueError("No linear solver specified.")

    def linearized_solve(self, a_assembled: BilinearForm, L_assembled: LinearForm, precond: Preconditioner, gfu: GridFunction) -> Tuple[float, float]:
        """
        Function to prepare for and perform the linear/linearized solve of the linear/nonlinear model.

        This function prepares for and solves the linear/linearized system associated with the model. This is included in the model class itself because, for nonlinear models, the linearization method(s) are model specific and are not
        generalized. For a linear model, this is called once in order to approximate the solution, but for a nonlinear
        model this function is repeatedly used by the nonlinear solver to approxiamte the solution.

        Args:
            a_lst: A list of the bilinear forms.
            L_lst: A list of the linear forms.
            precond_lst: A list of preconditioners to use.
            gfu: The gridfunction to store the solution of the linear system.

        Returns:
            Tuple of the error residual and norm of the solution.
        """
        # assume linear model, will be overriden for nonlinear models
        self.linear_solve(a_assembled, L_assembled, precond, gfu)

        # currently NGSolve's linear solver implementations do not expose the linear solver status, including residuals,
        # number of iterations to convergence, etc.
        # Assume that the linear solver converged and error is within numerical precision and the norm of the solution
        # is irrelevant
        return(0., 0.)

    def update_linearization(self, gfu: GridFunction):
        """
        Function to update variables needed to linearize the model (if nonlinear).

        This function updates variables used for linearization of nonlinear models, which is used by the solver class to
        construct a nonlinear solver.

        Args:
            gfu: The gridfunction to store the solution from the previous iteration or solution.
        """

        # assuming linear model
        pass
