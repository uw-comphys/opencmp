"""
Module for the base model class.
"""
import ngsolve as ngs
from config_functions import load_config, ConfigParser, BCFunctions, ICFunctions, ModelFunctions, RefSolFunctions
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from ngsolve.comp import ProxyFunction, FESpace, DifferentialSymbol, GridFunction
from diffuse_interface import DIM
from helpers.io import load_mesh


class Model(ABC):
    """
    Base class which other models will subclass from.
    """

    def __init__(self, config: ConfigParser, t_param: ngs.Parameter) -> None:
        """
        Initializer for the model classes

        Args:
            config: A configparser object loaded with config file
        """

        # Here for type-checking, do not remove.
        # Value must be set in the __init__ of a subclass, and must be set before calling super().
        self.model_components: Dict[str, Optional[int]]
        if not self.model_components:
            # This line needs to be here to satisfy mypy and Pycharm's checker
            # Otherwise it complains that Model has no attribute model_components.
            self.model_components = {}
            raise ValueError("Forgot to set self.model_components")

        # Now add any new model components. They get added to the mixed finite element space in the order in which they
        # were specified in the config file.
        new_components = config.get_list(['OTHER', 'component_names'], str, quiet=True)
        num_existing = len(self.model_components)
        for i in range(len(new_components)):
            self.model_components[new_components[i]] = i + num_existing

        # Also create a dictionary to hold the values of the model variables.
        self.model_components_values = self.model_components.copy()

        # Same as above.
        self.model_local_error_components: Dict[str, bool]
        if not self.model_local_error_components:
            self.model_local_error_components = {}
            raise ValueError("Forgot to set self.model_local_error_components")

        # Same as above.
        self.BC_init: Dict[str, Dict[str, Dict[str, ngs.CoefficientFunction]]]
        if not self.BC_init:
            self.BC_init = {}
            raise ValueError('Forgot to set self.BC_init.')

        # Set config file.
        self.config = config

        # Get the run directory.
        self.run_dir = self.config.get_item(['OTHER', 'run_dir'], str)

        # The name of the model (helper variable). It is the name of the class.
        self.name = self.__class__.__name__.lower()

        # Set time stepping parameter.
        self.t_param = t_param

        # Check if the diffuse interface method is being used and if yes initialize a DIM class object.
        self.DIM = self.config.get_item(['DIM', 'diffuse_interface_method'], bool, quiet=True)
        if self.DIM:
            self.DIM_dir = self.config.get_item(['DIM', 'dim_dir'], str)
            self.DIM_solver = DIM(self.DIM_dir)

        # Load the mesh. If the diffuse interface method is being used the mesh will be constructed/loaded by the DIM
        # solver.
        if self.DIM and (self.DIM_solver.load_method == 'generate' or self.DIM_solver.load_method == 'combine'):
            try:
                self.mesh = load_mesh(self.config) # TODO: Elizabeth this is a hack and needs to be fixed or it will blow up in your face.
            except:
                self.mesh = self.DIM_solver.mesh
        else:
            self.mesh = load_mesh(self.config)

        # Initialize classes to hold the functions for the model.
        self.bc_functions = BCFunctions(self.run_dir + '/bc_dir/config', self.t_param, self.model_components)
        self.ic_functions = ICFunctions(self.run_dir + '/ic_dir/config', self.t_param, self.model_components)
        self.model_functions = ModelFunctions(self.run_dir + '/model_dir/config', self.t_param, self.model_components)
        self.ref_sol_functions = RefSolFunctions(self.run_dir + '/ref_sol_dir/config', self.t_param, self.model_components)

        # Load the finite element space parameters.
        self.element = self.config.get_list(['FINITE ELEMENT SPACE', 'elements'], str)
        self.interp_ord = self.config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)

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
        self.nu = self.ipc * (self.interp_ord ** 2)

        # Load the solver parameters.
        self.solver = self.config.get_item(['SOLVER', 'solver'], str)
        if self.solver == 'default':
            # Default to a direct solve.
            self.solver = 'direct'

        self.preconditioner = self.config.get_item(['SOLVER', 'preconditioner'], str)
        if self.preconditioner == 'default':
            # 'local' seems to be the default preconditioner in the NGSolve examples.
            self.preconditioner = 'local'

        self.solver_tolerance = self.config.get_item(['SOLVER', 'solver_tolerance'], float, quiet=True)
        self.solver_max_iters = self.config.get_item(['SOLVER', 'solver_max_iterations'], int, quiet=True)

        self.verbose = self.config.get_item(['OTHER', 'messaging_level'], int, quiet=True) > 0

        # Note: Need dirichlet_names before constructing fes, but need to construct fes before gridfunctions can be
        # loaded for ex: BCs, ICs, reference solutions. So, get the BC dict before constructing fes, but then actually
        # load the BCs etc after constructing fes.
        #
        # Get the boundary conditions.
        self.BC_init, self.dirichlet_names = self.bc_functions.set_boundary_conditions(self.BC_init)

        # Create the finite element space.
        # This needs to be done before the initial conditions are loaded.
        self.fes = self._construct_fes()

        # Load initial condition.
        self.IC = self.construct_gfu()
        self.ic_functions.set_initial_conditions(self.IC, self.mesh, self.name, self.model_components)

        # Load any boundary conditions saved as gridfunctions.
        self.BC = self.bc_functions.load_bc_gridfunctions(self.BC_init, self.fes, self.model_components)
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu(), self.model_components)

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
            self.DIM_bc_functions = BCFunctions(self.DIM_dir + '/bc_dir/config', self.t_param, self.model_components)
            self.DIM_BC, DIM_dirichlet_names = self.DIM_bc_functions.set_boundary_conditions(self.BC_init)
            self.DIM_BC = self.DIM_bc_functions.load_bc_gridfunctions(self.DIM_BC, self.fes, self.model_components)

        # By default assume that the model does not need to be linearized.
        self.linearize = None

    def _construct_linearization_terms(self) -> Optional[List[ngs.GridFunction]]:
        """
        Function to construct the list of linearization terms.

        If a specific model does not need linearization terms, return None.

        Returns:
            ~: None or a list of grid functions used for linearizing a non-linear model
        """

        return None

    def _ds(self, marker: str) -> DifferentialSymbol:
        """
        Function to get the appropriate (DG vs non-DG) version of ds for the given marker.

        Args:
            marker: String representing the mesh marker to get ds on.

        Returns:
            ds: The appropriate ds.
        """
        if self.DG:
            ds = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(marker))
        else:
            ds = ngs.ds(definedon=self.mesh.Boundaries(marker))

        return ds

    def update_model_variables(self, updated_gfu: GridFunction, ic_update=False, ref_sol_update=False) -> None:
        """
        Function to update the values of the model variables and then re-parse any expressions that contain those model
        variables in ex: the model functions, the model boundary conditions...

        Args:
            updated_gfu: Gridfunction containing updated values for all model variables.
            ic_update: If True update the initial conditions (not clear when you would want to do this).
            ref_sol_update: If True update the reference solutions (not clear when you would want to do this).
        """

        # Update the values of the model variables.
        for key in self.model_components_values.keys():
            component = self.model_components[key]

            if component is None:
                # Gridfunction over the full finite element space.
                self.model_components_values[key] = updated_gfu
            else:
                # Component of the finite element space.
                self.model_components_values[key] = updated_gfu.components[component]

        # Re-parse any expressions as necessary with the new model variable values.
        #
        # Start with the boundary conditions.
        self.bc_functions.update_boundary_conditions(self.t_param, self.model_components_values)
        self.BC, dirichlet_names = self.bc_functions.set_boundary_conditions(self.BC)
        self.g_D = self.bc_functions.set_dirichlet_boundary_conditions(self.BC, self.mesh, self.construct_gfu(),
                                                                       self.model_components)

        # Also update the DIM boundary conditions if they exist.
        if self.DIM:
            self.DIM_bc_functions.update_boundary_conditions(self.t_param, self.model_components_values)
            self.DIM_BC, DIM_dirichlet_names = self.DIM_bc_functions.set_boundary_conditions(self.DIM_BC)

        # Update the model parameters and functions.
        self.model_functions.update_model_functions(self.t_param, self.model_components_values)
        self._set_model_parameters()

        if ic_update:
            # Update the initial conditions.
            self.ic_functions.update_initial_conditions(self.t_param, self.model_components_values)
            self.ic_functions.set_initial_conditions(self.IC, self.mesh, self.name, self.model_components)

        if ref_sol_update:
            # Update the reference solutions.
            self.ref_sol_functions.update_ref_solutions(self.t_param, self.model_components_values)
            self.ref_sol = self.ref_sol_functions.set_ref_solution(self.fes, self.model_components)

    def apply_dirichlet_bcs_to(self, gfu: ngs.GridFunction) -> None:
        """
        Function to set the Dirichlet boundary conditions within the solution GridFunction.

        Args:
            gfu: The GridFunction to add the Dirichlet boundary condition values to
        """
        # NOTE: DO NOT change from definedon=self.mesh.Boundaries(marker) to definedon=marker.
        if len(self.g_D) > 0:
            if len(gfu.components) == 0:  # Single trial functions
                # TODO: IDE is complaining that we don't specify parameter VOL_OR_BND for .Set()
                gfu.Set(self.g_D['u'], definedon=self.mesh.Boundaries(self.dirichlet_names['u']))
            else:  # Multiple trial functions.
                # TODO: There must be a better way to cycle through all of these, maybe with a lookup dictionary
                # Apply velocity BCs.
                component = self.model_components['u']
                gfu.components[component].Set(self.g_D['u'], definedon=self.mesh.Boundaries(self.dirichlet_names['u']))

                # Pressure BCs do not get applied.

    def construct_gfu(self) -> ngs.GridFunction:
        """
        Function to construct a solution GridFunction.

        Returns:
                ~: A GridFunction initialized on the finite element space of the model
        """
        gfu = ngs.GridFunction(self.fes)

        return gfu

    def construct_preconditioner(self, a_assembled: ngs.BilinearForm) -> ngs.Preconditioner:
        """
        Function to construct the preconditioner.

        Args:
            a_assembled: The Bilinear form.

        Returns:
            precond: The preconditioner.
        """
        precond = ngs.Preconditioner(a_assembled, self.preconditioner)

        return precond

    def construct_and_run_solver(self, a_assembled: ngs.BilinearForm, L_assembled: ngs.LinearForm, precond: ngs.Preconditioner, gfu: ngs.GridFunction) -> ngs.solvers:
        """
        Function to construct the solver.

        Args:
            a_assembled: The Bilinear form.
            precond: The preconditioner.

        Returns:
            inv: The solver.
        """

        if self.element[0] == 'HDiv' or self.element[0] == 'RT':
            # HDiv elements can only strongly apply u.n Dirichlet boundary conditions. In order to apply other Dirichlet
            # boundary conditions (ex: on the full vector or the tangential vector) the boundary conditions must be
            # weakly imposed. This is done in DG by penalization terms but requires solving over all DOFs, not just the
            # free DOFs. This will not work in CG, hence the warning to the user.
            no_constrained_dofs = self.config.get_item(['FINITE ELEMENT SPACE', 'no_constrained_dofs'], bool, quiet=True)
            if no_constrained_dofs:
                if not self.DG:
                    print('We strongly recommend using DG with HDiv if tangential Dirichlet boundary conditions need to '
                          'be applied.')

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

    def update_bcs(self, bc_dict_patch: Dict[str, Dict[str, Dict[str, Union[float, ngs.CoefficientFunction]]]]) -> None:
        """
        Function to update BCs to arbitrary values.

        This function is used to implement controllers since this is what lets them manipulate the manipulated variable.

        Args:
            bc_dict_patch: Dictionary containing new values for the BCs being updated
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

    def update_timestep(self, gfu: ngs.GridFunction, gfu_0: ngs.GridFunction) -> None:
        """
        Function to update the previous time-step GridFunction with result of the current time-step.

        Args:
            gfu: The GridFunction containing the result of the current time-step.
            gfu_0: The GridFunction containing the result of the previous time-step.
        """

        gfu_0.vec.data = gfu.vec

    @staticmethod
    @abstractmethod
    def allows_explicit_schemes() -> bool:
        """
        Function to specify whether a given model works with explicit time integration schemes.
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
            ~: The finite element space
        """

    @abstractmethod
    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        """
        Function return the trial and test (weighting) function(s) for the model.

        Returns:
            ~: Returns a the trial and test (weighting) function(s).
        """

    @abstractmethod
    def construct_bilinear(self, U: List[ProxyFunction], V: List[ProxyFunction],
                           explicit_bilinear: bool = False) -> ngs.BilinearForm:
        """
        Function to construct the bilinear form.

        Args:
            U: A list of trial functions for the model's finite element space
            V: A list of testing (weighting) functions for the model's finite element space
            W: List of GridFunction components used to manually linearize a non-linear problem
            explicit_bilinear: If True, the bilinear form is constructed in an explicit manner from only known
                               information.

        Returns:
            ~: The bilinear form for the model, without any of the time-discretization terms.
        """

    @abstractmethod
    def construct_linear(self, V: List[ProxyFunction], gfu_0: Union[GridFunction, None] = None) -> ngs.LinearForm:
        """
        Function to construct the linear form.

        Args:
            V: A list of testing (weighting) functions for the model's finite element space
            gfu_0: List of GridFunction components used to manually linearize a non-linear problem

        Returns:
            ~: The linear form for the model, without any of the time-discretization terms.
        """

    @abstractmethod
    def single_iteration(self, a: ngs.BilinearForm, L: ngs.LinearForm, precond: ngs.Preconditioner,
                         gfu: ngs.GridFunction) -> None:
        """
        Function to solve a single iteration (if time stepping) of the model.

        Args:
            a: The bilinear form
            L: The linear form
            precond: Preconditioner to use
            gfu: The GridFunction to store the solution form the current iteration

        Returns:

        """
