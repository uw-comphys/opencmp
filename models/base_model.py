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

import ngsolve as ngs
from config_functions import ConfigParser, BCFunctions, ICFunctions, ModelFunctions, RefSolFunctions
from config_functions.load_config import parse_str
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, cast, Sequence
from ngsolve.comp import ProxyFunction, FESpace, DifferentialSymbol
from ngsolve import Parameter, GridFunction, BilinearForm, LinearForm, Preconditioner, CoefficientFunction
from diffuse_interface import DIM
from helpers.io import load_mesh

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

        # Set time stepping parameter.
        self.t_param = t_param

        # Set config file.
        self.config = config

        # Get the run directory.
        self.run_dir = self.config.get_item(['OTHER', 'run_dir'], str)

        # Dictionary to map variable name to index inside fes/gridfunctions
        # Also used for labeling variables in vtk output.
        # NOTE: Here for type-checking, do not remove.
        #       Value must be set in the __init__ of a subclass, and must be set before calling super().
        self.model_components: Dict[str, Optional[int]]
        if not self.model_components:
            # This line needs to be here to satisfy mypy and Pycharm's checker
            # Otherwise it complains that Model has no attribute model_components.
            self.model_components = {}
            raise ValueError('Forgot to set self.model_components')

        # Create a list of dictionaries to hold the values of any model variables or parameters that are variables in
        # model function definitions (ex: if source = u * diffusion_coefficients_a then this list of dictionaries holds
        # the values of u and diffusion_coefficients_a at each time step in the time discretization scheme).
        self.update_variables = [self.model_components.copy() for _ in self.t_param]
        self._add_multiple_parameters(config)

        # Dictionary to specify if a particular component should be used for error calculations
        # NOTE: Here for type-checking, do not remove.
        #       Value must be set in the __init__ of a subclass, and must be set before calling super().
        self.model_local_error_components: Dict[str, bool]
        if not self.model_local_error_components:
            self.model_local_error_components = {}
            raise ValueError('Forgot to set self.model_local_error_components')

        # List of dictionaries to specify if a particular component should have a time derivative.
        # Each dictionary in the list corresponds to a different weak form for the model.
        # NOTE: Here for type-checking, do not remove.
        #       Value must be set in the __init__ of a subclass, and must be set before calling super().
        self.time_derivative_components: List[Dict[str, bool]]
        if not self.time_derivative_components:
            self.time_derivative_components = []
            raise ValueError('Forgot to set self.time_derivative_components.')

        # The number of weak forms used by the model. E.g. the drift-flux model has two separate weak forms
        # due to its discretization scheme. conservation of mass and momentum as de-coupled and solved sequentially
        # thus each needs an independent weak form.
        # NOTE: Here for type-checking, do not remove.
        #       Value must be set in the __init__ of a subclass, and must be set before calling super().
        self.num_weak_forms: int
        if not self.num_weak_forms:
            self.num_weak_forms = 0
            raise ValueError('Forgot to set self.num_weak_forms.')

        # Ensure that local error calculations are specified for each component
        assert len(self.model_components)\
               == len(self.model_local_error_components)
        # Ensure that each weak form has derivatives associated with it
        assert self.num_weak_forms == len(self.time_derivative_components)
        # Ensure that every variable is specified for the time derivative terms for each weak form
        for form in self.time_derivative_components:
            assert len(self.model_components) == len(form)

        # NOTE: Here for type-checking, do not remove.
        #       Value must be set in the __init__ of a subclass, and must be set before calling super().
        self.BC_init: Dict[str, Dict[str, Dict[str, ngs.CoefficientFunction]]]
        if not self.BC_init:
            self.BC_init = {}
            raise ValueError('Forgot to set self.BC_init.')

        # Invert the model components dictionary, so that you can loop up component name by index
        self.model_components_inverted = dict(zip(self.model_components.values(), self.model_components.keys()))

        # The name of the model (helper variable). It is the name of the class.
        self.name = self.__class__.__name__.lower()

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
        assert len(self.model_components) == len(self.element)

        # Check if DG should be used.
        self.DG = self.config.get_item(['DG', 'DG'], bool)

        if self.DG:
            # Load the DG parameters.
            self.ipc = self.config.get_item(['DG', 'interior_penalty_coefficient'], float)
        else:
            # Need to define this parameter, but won't end up using it so suppress the default value warning.
            self.ipc = self.config.get_item(['DG', 'interior_penalty_coefficient'], float, quiet=True)

        # TODO: we should probably rename self.nu to something else so it doesnt conflict with kinematic viscosity
        # Construct nu as ipc*interp_ord^2.
        self.nu = self.ipc * self.interp_ord ** 2

        # Load the solver parameters.
        self.solver = self.config.get_item(['SOLVER', 'solver'], str)
        if self.solver == 'default':
            # Default to a direct solve.
            self.solver = 'direct'

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
                if self.solver in ['CG', 'MinRes']:
                    # CG and MinRes require a preconditioner or they fail with an AssertionError.
                    raise ValueError('Preconditioner can\'t be None if using CG or MinRes solvers.')

        self.solver_tolerance = self.config.get_item(['SOLVER', 'solver_tolerance'], float, quiet=True)
        self.solver_max_iters = self.config.get_item(['SOLVER', 'solver_max_iterations'], int, quiet=True)

        self.verbose = self.config.get_item(['OTHER', 'messaging_level'], int, quiet=True) > 0

        # Initialize the BC functions class
        # Note: Need dirichlet_names before constructing fes, but need to construct fes before gridfunctions can be
        #       loaded for ex: BCs, ICs, reference solutions. So, get the BC dict before constructing fes,
        #       but then actually load the BCs etc after constructing fes.
        self.bc_functions       = BCFunctions(self.run_dir + '/bc_dir/bc_config', self.run_dir, self.mesh, self.t_param,
                                              self.update_variables)

        # Get the boundary conditions.
        self.BC, self.dirichlet_names = self.bc_functions.set_boundary_conditions(self.BC_init)

        # Create the finite element space.
        # This needs to be done before the initial conditions are loaded.
        self.fes = self._construct_fes()

        self._test, self._trial = self._create_test_and_trial_functions()

        # Create the rest of the ConfigFunction classes
        self.ic_functions       = ICFunctions(self.run_dir + '/ic_dir/ic_config', self.run_dir, self.mesh, self.t_param,
                                              self.update_variables)
        self.model_functions    = ModelFunctions(self.run_dir + '/model_dir/model_config', self.run_dir, self.mesh,
                                                 self.t_param, self.update_variables)
        self.ref_sol_functions  = RefSolFunctions(self.run_dir + '/ref_sol_dir/ref_sol_config', self.run_dir, self.mesh,
                                                  self.t_param, self.update_variables)

        # Load initial condition.
        self.IC = self.construct_gfu()
        self.ic_functions.set_initial_conditions(self.IC, self.mesh, self.name, self.model_components)

        # Load any boundary conditions saved as gridfunctions.
        self.BC = self.bc_functions.load_bc_gridfunctions(self.BC, self.fes, self.model_components)
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu(),
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
            self.DIM_bc_functions = BCFunctions(self.DIM_dir + '/bc_dir/dim_bc_config', self.run_dir, None, self.t_param,
                                                self.update_variables)
            self.DIM_BC, DIM_dirichlet_names = self.DIM_bc_functions.set_boundary_conditions(self.BC_init)
            self.DIM_BC = self.DIM_bc_functions.load_bc_gridfunctions(self.DIM_BC, self.fes, self.model_components)

        # By default assume that the model does not need to be linearized.
        self.linearize = None

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
            assert param not in self.update_variables[0].keys() # Should never be using the same name for a model component and parameter.
            for time_dict in self.update_variables:
                time_dict[param] = 0

    def _create_test_and_trial_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        """
        Function to create the trial and test (weighting) function(s) for the model.

        Returns:
            ~: Returns two lists, one for the trial and test (weighting) function(s).
        """
        # Get test and trial functions
        trial = self.fes.TrialFunction()
        test  = self.fes.TestFunction()

        # For single variable models, trial and test will be a single value, we want a list
        if type(trial) is not list:
            trial = [trial]
        if type(test) is not list:
            test  = [test]

        return test, trial

    def _construct_linearization_terms(self) -> Optional[List[GridFunction]]:
        """
        Function to construct the list of linearization terms.

        If a specific model does not need linearization terms, return None.

        Returns:
            None or a list of grid functions used to linearize a non-linear model.
        """

        return None

    def update_linearization_terms(self, gfu: GridFunction) -> None:
        """
        Function to update the values in the linearization terms.

        Required for Runge Kutta-type solvers if the model's weak form involves linearization terms. Assumes that
        linearization terms (self.W) have already been created for the model.

        Args:
            gfu: The gridfunction to use to update the linearization terms.
        """

        # Most models do not have nonlinear terms (or use IMEX-type linearization) so default to doing nothing.
        return None

    def _ds(self, marker: str) -> DifferentialSymbol:
        """
        Function to get the appropriate (DG vs non-DG) version of ds for the given marker.

        Args:
            marker: String representing the mesh marker to get ds on.

        Returns:
            The appropriate ds.
        """
        if self.DG:
            ds = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(marker))
        else:
            ds = ngs.ds(definedon=self.mesh.Boundaries(marker))

        return ds

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
            self.IC = self.construct_gfu()
            self.ic_functions.set_initial_conditions(self.IC, self.mesh, self.name, self.model_components)

        if ref_sol_update:
            # Update the reference solutions.
            self.ref_sol_functions.update_ref_solutions(self.t_param, self.update_variables, self.mesh)
            self.ref_sol = self.ref_sol_functions.set_ref_solution(self.fes, self.model_components)

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
                    # Apply Dirichlet or pinned BCs.
                    i = self.model_components[component_name]
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

    def construct_preconditioners(self, a_assembled: List[BilinearForm]) -> List[Preconditioner]:
        """
        Function to construct the preconditioners needed by the model.

        Args:
            a_assembled_lst: A list of the assembled bilinear form.

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

    def construct_and_run_solver(self, a_assembled: BilinearForm, L_assembled: LinearForm, precond: Preconditioner,
                                 gfu: GridFunction):
        """
        Function to construct the solver and run one solve on the provided gridfunction.

        Args:
            a_assembled: The assembled bilinear form.
            L_assembled: The assembled linear form.
            precond: The preconditioner.
            gfu: The gridfunction holding information about any Dirichlet BCs which will be updated to hold the
                solution.
        """

        if precond is None:
            # Need to provide a freedofs argument to the solver. Confirm that the user hasn't specified that all
            # dofs should be free dofs.
            no_constrained_dofs = self.config.get_item(['FINITE ELEMENT SPACE', 'no_constrained_dofs'], bool, quiet=True)
            if no_constrained_dofs:
                raise ValueError('Must constrain Dirichlet DOFs if not providing a preconditioner.')

            freedofs = self.fes.FreeDofs()

        else:
            if 'HDiv' in self.element.values() or 'RT' in self.element.values():
                # HDiv elements can only strongly apply u.n Dirichlet boundary conditions. In order to apply other
                # Dirichlet boundary conditions (ex: on the full vector or the tangential vector) the boundary
                # conditions must be weakly imposed. This is done in DG by penalization terms but requires solving
                # over all DOFs, not just the free DOFs. This will not work in CG, hence the warning to the user.
                no_constrained_dofs = self.config.get_item(['FINITE ELEMENT SPACE', 'no_constrained_dofs'], bool, quiet=True)
                if no_constrained_dofs:
                    if not self.DG:
                        print('We strongly recommend using DG with HDiv if tangential Dirichlet boundary conditions' 
                              'need to be applied.')

                    freedofs = None
                else:
                    freedofs = self.fes.FreeDofs()
            else:
                freedofs = self.fes.FreeDofs()


        if self.solver == 'direct':
            inv = a_assembled.mat.Inverse(freedofs=freedofs)
            r = L_assembled.vec.CreateVector()
            r.data = L_assembled.vec - a_assembled.mat * gfu.vec
            gfu.vec.data += inv * r

        elif self.solver == 'CG':
            ngs.solvers.CG(mat=a_assembled.mat, rhs=L_assembled.vec, pre=precond, sol=gfu.vec,
                           tol=self.solver_tolerance, maxsteps=self.solver_max_iters, printrates=self.verbose,
                           initialize=False)

        elif self.solver == 'MinRes':
            ngs.solvers.MinRes(mat=a_assembled.mat, rhs=L_assembled.vec, pre=precond, sol=gfu.vec,
                               tol=self.solver_tolerance, maxsteps=self.solver_max_iters,
                               printrates=self.verbose)

        elif self.solver == 'GMRes':
            ngs.solvers.GMRes(A=a_assembled.mat, b=L_assembled.vec, pre=precond, freedofs=freedofs,
                              x=gfu.vec, tol=self.solver_tolerance, maxsteps=self.solver_max_iters,
                              printrates=self.verbose)

        elif self.solver == 'Richardson':
            # TODO: User should be able to set a damping factor.
            ret = ngs.solvers.PreconditionedRichardson(a=a_assembled, rhs=L_assembled.vec, pre=precond,
                                                       freedofs=freedofs, tol=self.solver_tolerance,
                                                       maxit=self.solver_max_iters, printing=self.verbose)
            gfu.vec.data = ret

    # TODO: Move to time_integration_schemes.py
    def time_derivative_terms(self, gfu_lst: List[List[GridFunction]], scheme: str, step: int = 1)\
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
                        a_tmp += (11/6) * U[i] * V[i]
                        L_tmp += (3 * gfu_lst[1][i] - 1.5 * gfu_lst[2][i] + 1/3 * gfu_lst[3][i]) * V[i]
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

    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        """
        Function return the trial and test (weighting) function(s) for the model.

        Returns:
            ~: Returns two lists, one for the trial and test (weighting) function(s).
        """
        return self._trial, self._test

    def update_bcs(self, bc_dict_patch: Dict[str, Dict[str, Dict[str, Union[float, CoefficientFunction]]]]) -> None:
        """
        Function to update BCs to arbitrary values.

        This function is used to implement controllers. It lets them manipulate the manipulated variable.

        Args:
            bc_dict_patch: Dictionary containing new values for the BCs being updated.
        """
        # Check that the specified BC is valid
        for bc_type in bc_dict_patch.keys():
            for var_name in bc_dict_patch[bc_type].keys():
                for bc_location in bc_dict_patch[bc_type][var_name].keys():
                    if self.BC.get(bc_type, {}).get(var_name, {}).get(bc_location, None) is None:
                        # One or more of the provided bcs is not formated correctly
                        raise ValueError('BC type \'{}\' for variable \'{}\' at location \'{}\' does not exist'.
                                         format(bc_type, var_name, bc_location))
                    else:
                        # Set the new BC value
                        self.BC[bc_type][var_name][bc_location] = bc_dict_patch[bc_type][var_name][bc_location]

        # Reload everything as grid functions
        self.BC = self.bc_functions.load_bc_gridfunctions(self.BC, self.fes, self.model_components)
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu(), self.model_components)

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
    def _set_model_parameters(self) -> None:
        """
        Function to initialize model parameters and functions from the configfile.
        """

    @abstractmethod
    def _construct_fes(self) -> FESpace:
        """
        Function to construct the finite element space for the model.

        Returns:
            The finite element space.
        """

    @abstractmethod
    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        """
        Function to construct a portion of the bilinear form for model variables WITH time derivaties.

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
    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:
        """
        Function to construct a portion of the bilinear form for model variables WITHOUT time derivaties.

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
    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                         dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:
        """
        Function to construct the linear form.

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
    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]] = None,
                                dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[LinearForm]:
        """
        Function to construct the explicit terms used in an IMEX linearization.

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
    def single_iteration(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                         precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:
        """
        Function to solve a single iteration (if time stepping) of the model.

        Args:
            a_lst: A list of the bilinear forms.
            L_lst: A list of the linear forms.
            precond_lst: A list of preconditioners to use.
            gfu: The gridfunction to store the solution from the current iteration.
            time_step: What time step values to use if _apply_dirichlet_bcs_to must be called.
        """
