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

from __future__ import annotations
from ..models import Model
import ngsolve as ngs
from ngsolve import CoefficientFunction, GridFunction, Preconditioner
import math
from typing import Dict, List, Optional, Tuple, Type
from abc import ABC, abstractmethod
from ..config_functions import ConfigParser
import sys
from ..helpers.saving import SolutionFileSaver
from ..helpers.error import calc_error
from ..helpers.ngsolve_ import gridfunction_rigid_body_motion
import numpy as np
from pathlib import Path

from ..controllers.controller_group import ControllerGroup

"""
Module for the base solver class.
"""

# Dictionary holding the time discretization scheme name and either the number of previous time steps it uses if a
# multistep scheme or the number of intermediate steps it takes if a Runge Kutta scheme.
scheme_order = {'explicit euler': 1,
                'implicit euler': 1,
                'crank nicolson': 1,
                'adaptive two step': 1,
                'adaptive three step': 2,
                'euler IMEX': 1,
                'CNLF': 2,
                'SBDF': 3,
                'adaptive IMEX': 3,
                'RK 222': 2,
                'RK 232': 3}

# Dictionary holding the time step coefficients. This is only relevant for Runge Kutta schemes where the intermediate
# steps use modified dt values.
scheme_dt_coef = {'explicit euler': [1.0],
                  'implicit euler': [1.0],
                  'crank nicolson': [1.0],
                  'adaptive two step': [1.0],
                  'adaptive three step': [1.0, 0.5],
                  'euler IMEX': [1.0],
                  'CNLF': [1.0, 1.0],
                  'SBDF': [1.0, 1.0, 1.0],
                  'adaptive IMEX': [1.0, 1.0, 1.0],
                  'RK 222': [1.0, 0.5 * (2.0 - ngs.sqrt(2.0))],
                  'RK 232': [1.0, 1.0, 0.5 * (2.0 - ngs.sqrt(2.0))]}


class Solver(ABC):
    """
    Base class for the different stationary/transient solvers.
    """
    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        """
        This function initializes the TimeSolver class.

        Args:
            model_class: The model to solve.
        """
        self.config = config

        self.transient = self.config.get_item(['TRANSIENT', 'transient'], bool)

        self.gfu_0_list: List[ngs.GridFunction] = []

        if self.transient:
            self.scheme = self.config.get_item(['TRANSIENT', 'scheme'], str)
            self.scheme_order = scheme_order[self.scheme]
            self.scheme_dt_coef = scheme_dt_coef[self.scheme]
            # Differentiate between multi-step and Runge-Kutta schemes. They use time step information differently.
            if self.scheme in ['RK 222', 'RK 232', 'adaptive three step']:
                self.scheme_type = 'RK'
            else:
                self.scheme_type = 'multistep'

            self.t_range = self.config.get_list(['TRANSIENT', 'time_range'], float)

            # Initialize the time step from the config file. Then generate a list of time steps and a list of time
            # parameters for the current time step and each previous time step/intermediate step that will be used in
            # the time discretization scheme. These lists are taken to be in reverse chronological order.
            # Ex: implicit Euler, a single step multistep scheme would end up with:
            #       dt_param = [t_0-t_-1, t_-1-t_-2]
            #       t_param = [t_0, t_-1]
            # Ex: A three-step Runge Kutta scheme would end up with:
            #       dt_param = [t_0-t_-1, t^2-t_-1, t^1-t_-1, t_-1-t_-2]
            #       t_param = [t_0, t^2, t^1, t_-1]
            self.dt_param_init = ngs.Parameter(self.config.get_item(['TRANSIENT', 'dt'], float))

            self.dt_param = [ngs.Parameter(self.dt_param_init.Get() * self.scheme_dt_coef[i]) for i in range(self.scheme_order)]
            self.dt_param.append(ngs.Parameter(self.dt_param_init.Get() * self.scheme_dt_coef[0]))

            self.t_param = [ngs.Parameter(self.t_range[0])]
            if self.scheme_type == 'RK':
                self.t_param += [ngs.Parameter(self.t_range[0] - self.dt_param[0].Get() + self.dt_param[i].Get()) for i in range(1, self.scheme_order)]
                self.t_param.append(ngs.Parameter(self.t_range[0] - self.dt_param[0].Get()))
            else:
                for i in range(self.scheme_order):
                    tmp = self.t_range[0]
                    for j in range(i + 1):
                        tmp -= self.dt_param[j].Get()
                    self.t_param.append(ngs.Parameter(tmp))

            self.has_controller = self.config.get_item(['CONTROLLER', 'active'], bool)

            if 'adaptive' in self.scheme:
                self.adaptive = True

                # An adaptive scheme has additional parameters.
                dt_tol = self.config.get_dict(['TRANSIENT', 'dt_tolerance'], '', None)
                self.dt_abs_tol = dt_tol['absolute']
                self.dt_rel_tol = dt_tol['relative']

                self.dt_range = self.config.get_list(['TRANSIENT', 'dt_range'], float)
                if self.dt_range[0] > self.dt_range[1]:
                    # Make sure dt_range is in the order [min_dt, max_dt]
                    self.dt_range = [self.dt_range[1], self.dt_range[0]]
                self.max_rejects = self.config.get_item(['TRANSIENT', 'maximum_rejected_solves'], int, quiet=True)
            else:
                self.adaptive = False

            if self.scheme == 'explicit euler' and not model_class.allows_explicit_schemes():
                # Check that the time integration scheme is implicit. An explicit scheme will give a singular matrix.
                raise ValueError('Only implicit time integration schemes can be used with Stokes and INS')

            # TODO: Import controller
        else:
            self.t_param = [ngs.Parameter(0.0)]
            self.scheme_type = 'stationary'

        # Initialize model
        self.model = model_class(self.config, self.t_param)

        # Set everything up for if error metrics should be calculated and saved to file after every time step.
        self.check_error = self.config.get_item(['ERROR ANALYSIS', 'check_error_every_timestep'], bool, quiet=True)
        self.save_error = self.config.get_item(['ERROR ANALYSIS', 'save_error_every_timestep'], bool, quiet=True)

        if self.save_error:
            # Error values at each time step will be saved to a file inside the output directory. Need to make sure the
            # output directory exists or create it if it doesn't exist.
            Path(self.model.run_dir + '/output/').mkdir(parents=True, exist_ok=True)
            self.save_error_filename = self.model.run_dir + '/output/error_at_each_timestep.txt'

            with open(self.save_error_filename, 'w') as f:
                header = [metric + '_' + var for metric, var_lst in self.model.ref_sol['metrics'].items() for var in var_lst]
                header.insert(0, 'time')
                f.write(', '.join(header) + '\n')

        # This needs to be here since it needs a model, and the model needs the t_param
        if self.transient and self.has_controller:
            self.controller_group = ControllerGroup(self.t_param, self.model, self.config)

        # Check that the linearization method is consistent with the time integration scheme chosen.
        if self.transient:
            if self.scheme in ['euler IMEX', 'CNLF', 'SBDF', 'adaptive IMEX', 'RK 222', 'RK 232']:
                if self.model.linearize == 'Oseen':
                    # IMEX + using Oseen linearization is nonsensical.
                    raise ValueError('Oseen linearization can\'t be used with IMEX time integration schemes.')
            elif self.model.linearize == 'IMEX':
                # Not an IMEX scheme (would be caught by previous if statement) + not using IMEX linearization is
                # nonsensical.
                raise ValueError('IMEX linearization must be used with an IMEX time integration scheme.')

        self.gfu = self.model.construct_gfu()

        self._load_and_apply_initial_conditions()

        # Number of timesteps takes
        self.num_iters = 0
        # Flag to prevent an infinite loop of rejected runs.
        self.num_rejects = 0

        self.U, self.V = self.model.get_trial_and_test_functions()

        # Update the model component values for time step t^n+1 to be the trial functions. If a Runge Kutta scheme is
        # being used do this for every single time step since all time steps act as the to-be-solved time step over the
        # course of a single time step solve.
        if self.scheme_type == 'RK':
            for i in range(len(self.t_param)-1):
                self.model.update_model_variables(self.U, time_step=i)
        else:
            self.model.update_model_variables(self.U, time_step=0)

        self.saver: Optional[SolutionFileSaver] = None

        self.save_to_file = self.config.get_item(['VISUALIZATION', 'save_to_file'], bool)

        if self.save_to_file:
            self.saver = SolutionFileSaver(self.model, quiet=True)
            save_freq = self.config.get_list(['VISUALIZATION', 'save_frequency'], str, quiet=True)
            self.save_freq = [float(save_freq[0]), save_freq[1]]

            if self.save_freq[1] not in ['time', 'numit']:
                raise ValueError('Save frequency must be specified either per amount of time (\"time\") '
                                 'or per number of iterations (\"numit\").')

        # If there is an initial condition, save it.
        if self.save_to_file:
            if hasattr(self, 'gfu_0_list'):
                self.saver.save(self.gfu_0_list[0], self.t_param[0].Get())

            if self.model.DIM:
                self.saver.save(self.model.DIM_solver.phi_gfu_orig, self.t_param[0].Get(), DIM=True)

    def _dt_for_next_time_to_hit(self) -> float:
        """
        Function that calculate the time step till the next time that the model MUST be solved at due to a variety of
        constraints.

        This time is used to set the time step in order to ensure that the model is solved at this time.

        Returns:
            The next time that the model MUST be solved at.
        """
        times: List[float] = []

        # Add end time point to ensure we stop exactly at it
        times.append(self.t_range[1])

        # Check that the new dt won't cause the next solution save time to be skipped
        if self.save_to_file and (self.save_freq[1] == 'time'):
            # If simulation started at a time other than 0, shift the time for this calculation so that it starts at 0
            corrected_time          = self.t_param[0].Get() - self.t_range[0]
            # The number of save periods (rounded to next next highest)
            n_save_periods          = math.ceil(corrected_time / self.save_freq[0])
            # Time for the 1st next save. Need to re-account for the simulation possibly not starting at 0.
            time_for_next_save      = n_save_periods * self.save_freq[0] + self.t_range[0]
            # Time for the 2nd after save. Need to re-account for the simulation possibly not starting at 0.
            time_for_next_next_save = (n_save_periods + 1) * self.save_freq[0] + self.t_range[0]

            times.append(time_for_next_save)
            times.append(time_for_next_next_save)

        # Add the next time from the controller
        if self.has_controller:
            times += self.controller_group.get_time_for_next_two_control_actions()

        dts = np.array(times) - self.t_param[0].Get()

        dts_pos = dts[dts > 0]

        if len(dts_pos) == 0:
            # It doesn't matter what we return at this point. The only time this statement is reached is after the
            # final timestep is taken.
            return 42
        else:
            return min(dts_pos)

    def _log_timestep(self, accepted: bool, error_abs: float, error_rel: float, component: str) -> None:
        """
        Function to print out information about the current timestep's iteration

        Args:
            accepted: Boolean indicating if the current timestep was accepted, or if it being rerun with a smaller dt.
            error_abs: The absolute local error for this timestep. (not absolute value, but raw error value)
            error_rel: The relative local error for this timestep.
        """
        if accepted:
            verb = 'Keeping'
        else:
            verb = 'Rejecting'

        print('')
        print('Current time: {}'.format(self.t_param[0].Get()))
        # Simple Transient solve returns a negative error to match function signature, but it is meaningless
        if error_abs >= 0:
            print('{} solve.'.format(verb))
            print('Max error from {}'.format(component))
            print('Max REL error: {}'.format(error_rel))
            print('Max ABS error: {}'.format(error_abs))
        print('New dt:       {}'.format(self.dt_param[0].Get()))
        print('---')

    def _solve(self) -> None:
        """
        Function to perform a single iteration of the solve.
        """

        # Loop until an iteration is accepted, or until the max number of attempts is reached
        while True:
            if self.transient:
                # Update time. The calculation for the new dt ensures we don't overshoot the final time.
                # The new time step (dt) gets updated at the end of the solve (in the case of adaptive time-stepping,
                # otherwise it is constant).
                if self.scheme_type == 'RK':
                    for i in range(len(self.t_param)):
                        self.t_param[i].Set(self.t_param[i].Get() + self.dt_param[0].Get())
                else:
                    for i in range(len(self.t_param)):
                        self.t_param[i].Set(self.t_param[i].Get() + self.dt_param[i].Get())

                if self.model.DIM and self.model.DIM_solver.rigid_body_motion:
                    # If using rigid body motion need to update the phase field to reflect the new location of the
                    # phase field.
                    gridfunction_rigid_body_motion(self.t_param[0], self.model.DIM_solver.phi_gfu_orig,
                                                   self.model.DIM_solver.phi_gfu, self.model.DIM_solver.inv_R,
                                                   self.model.mesh, self.model.DIM_solver.N,
                                                   self.model.DIM_solver.scale, self.model.DIM_solver.offset)

            self._apply_boundary_conditions()

            self._re_assemble()

            self._single_solve()

            # Calculate local error, accept/reject current result, and update timestep
            accept_this_iteration, local_error_abs, local_error_rel, component = self._update_time_step()

            # Log information about the current timestep
            self._log_timestep(accept_this_iteration, local_error_abs, local_error_rel, component)

            # If this iteration met all all requirements for accepting the solution
            if accept_this_iteration:
                # Reset counter
                self.num_rejects = 0
                # Increment
                self.num_iters += 1

                if self.save_to_file and self.transient:
                    if self.save_freq[1] == 'time':
                        # This is ugly, but it's needed to make the modulo math work.
                        # 0.9999999 % 1 = 0.999999 but 1.000001 % 1 = 0 as expected.
                        if self.save_freq[0] < 1.0:
                            tmp_delta = (self.t_param[0].Get() - self.t_range[0]) * (1/self.save_freq[0])
                            if tmp_delta < 1.0:
                                tmp = tmp_delta - 1.0
                            else:
                                tmp = tmp_delta % 1.0
                        else:
                            tmp = (self.t_param[0].Get() - self.t_range[0]) % self.save_freq[0]
                        if math.isclose(tmp, 0.0, abs_tol=self.dt_param[0].Get() * 1e-2):
                            self.saver.save(self.gfu, self.t_param[0].Get())

                            if self.model.DIM:
                                self.saver.save(self.model.DIM_solver.phi_gfu, self.t_param[0].Get(), DIM=True)
                    elif self.save_freq[1] == 'numit':
                        if self.num_iters % self.save_freq[0] == 0:
                            self.saver.save(self.gfu, self.t_param[0].Get())

                            if self.model.DIM:
                                self.saver.save(self.model.DIM_solver.phi_gfu, self.t_param[0].Get(), DIM=True)

                # This iteration was accepted, break out of the while loop
                break
            else:
                self.num_rejects += 1

                # Prevent an infinite loop of rejections by ending the run.
                if self.num_rejects > self.max_rejects:
                    # Save the current solution before ending the run.
                    if self.save_to_file:
                        self.saver.save(self.gfu, self.t_param[0].Get())

                        if self.model.DIM:
                            self.saver.save(self.model.DIM_solver.phi_gfu, self.t_param[0].Get(), DIM=True)
                    else:
                        tmp_saver = SolutionFileSaver(self.model, quiet=True)
                        tmp_saver.save(self.gfu, self.t_param[0].Get())

                        if self.model.DIM:
                            tmp_saver.save(self.model.DIM_solver.phi_gfu, self.t_param[0].Get(), DIM=True)

                    sys.exit('At t = {0} the maximum number of rejected time steps has been exceeded. Saving current '
                             'solution to file and ending the run.'.format(self.t_param[0].Get()))

    def _update_bcs(self, bc_dict_patch: Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]]) -> None:
        """
        Function to update the model's BCs to arbitrary values, and then recreate the linear/bilinear forms and the
        preconditioner.

        This function is used by controllers in order to act on the manipulated variables.

        Args:
            bc_dict_patch: Dictionary containing new values for the BCs being updated.
        """
        # Update the model's BCs
        self.model.update_bcs(bc_dict_patch)

        # Recreate the linear form, bilinear form, and preconditioner
        self._create_linear_and_bilinear_forms()
        self._assemble()
        self._create_preconditioners()

    def reset_model(self) -> None:
        """
        Function to reset certain model variables back to an initial state.

        Needed for running convergence tests.
        """
        if self.transient:
            self.t_range = self.config.get_list(['TRANSIENT', 'time_range'], float)

            self.dt_param = [ngs.Parameter(self.dt_param_init.Get() * self.scheme_dt_coef[i]) for i in range(self.scheme_order)]
            self.dt_param.append(ngs.Parameter(self.dt_param_init.Get() * self.scheme_dt_coef[0]))
            self.t_param[0].Set(self.t_range[0])
            if self.scheme_type == 'RK':
                for i in range(1, self.scheme_order):
                    self.t_param[i].Set(self.t_range[0] - self.dt_param[0].Get() + self.dt_param[i].Get())
                self.t_param[-1].Set(self.t_range[0] - self.dt_param[0].Get())
            else:
                for i in range(self.scheme_order):
                    tmp = self.t_range[0]
                    for j in range(i + 1):
                        tmp -= self.dt_param[j].Get()
                    self.t_param[i+1].Set(tmp)

            if self.has_controller:
                self.controller_group = ControllerGroup(self.t_param, self.model, self.config)

            if self.adaptive:
                dt_tol = self.config.get_dict(['TRANSIENT', 'dt_tolerance'], '', None)
                self.dt_abs_tol = dt_tol['absolute']
                self.dt_rel_tol = dt_tol['relative']
                self.dt_range = self.config.get_list(['TRANSIENT', 'dt_range'], float)
                if self.dt_range[0] > self.dt_range[1]:
                    # Make sure dt_range is in the order [min_dt, max_dt]
                    self.dt_range = [self.dt_range[1], self.dt_range[0]]
            else:
                self.scheme = self.config.get_item(['TRANSIENT', 'scheme'], str)

        else:
            self.t_param = [ngs.Parameter(0.0)]
            self.scheme_type = 'stationary'

        # If error metrics are being saved after every time step add a note to the file that the model was reset.
        if self.save_error:
            with open(self.save_error_filename, 'a') as f:
                f.write('reset model\n')

        self.gfu = self.model.construct_gfu()

        if self.transient:
            # If a transient solve is being conducted the model variables need to be reset to the initial condition. The
            # initial condition and reference solution may also need to be reloaded if the mesh or finite element space
            # has changed.
            self.model.update_model_variables(self.gfu, ic_update=True, ref_sol_update=True, time_step=None)

        self._load_and_apply_initial_conditions()

        self.model.update_linearization_terms(self.model.IC)

        self.num_iters = 0
        self.num_rejects = 0  # Flag to prevent an infinite loop of rejected runs.

        self.U, self.V = self.model.get_trial_and_test_functions()

        # Update the model component values for time step t^n+1 to be the trial functions. If a Runge Kutta scheme is
        # being used do this for every single time step since all time steps act as the to-be-solved time step over the
        # course of a single time step solve.
        if self.scheme_type == 'RK':
            for i in range(len(self.t_param) - 1):
                self.model.update_model_variables(self.U, time_step=i)
        else:
            self.model.update_model_variables(self.U, time_step=0)

        # If there is an initial condition, save it.
        if self.save_to_file:
            if hasattr(self, 'gfu_0_list'):
                self.saver.save(self.gfu_0_list[0], self.t_param[0].Get())

            if self.model.DIM:
                self.saver.save(self.model.DIM_solver.phi_gfu_orig, self.t_param[0].Get(), DIM=True)

    def solve(self) -> GridFunction:
        """
        This function solves the model, either a single stationary solve or the entire transient solve.

        Returns:
            GridFunction containing the solution. For a transient solve this will be the result of the final time step.
        """
        self._create_linear_and_bilinear_forms()

        self._create_preconditioners()

        self._assemble()

        # Directly after initialization all elements of gfu_0_list contain the initial condition.
        self.gfu.vec.data = self.gfu_0_list[0].vec

        if self.transient:
            # Iterate over time steps.
            # NOTE: The first part of the and is somewhat redundant, but it ensures we don't go beyond the final time.
            while (self.t_param[0].Get() < self.t_range[1]) and not np.isclose(self.t_param[0].Get(), self.t_range[1]):
                # If there are controllers, calculate their control action
                # This runs BEFORE _solve so that it also calculates a control action based on the IC
                if self.has_controller:
                    control_bc_dict = self.controller_group.calculate_control_all_actions(self.gfu,
                                                                                          rk_scheme=self.scheme_type == "RK")
                    self._update_bcs(control_bc_dict)

                self._solve()

                if self.check_error and self.save_error:
                    # Print out the error metrics at each time step and save them to file.
                    error_lst = calc_error(self.config, self.model, self.gfu)
                    error_lst.insert(0, str(self.t_param[0].Get()))

                    # Write the calculated error metrics at the given time step to file.
                    with open(self.save_error_filename, 'a') as f:
                        f.write(', '.join([str(item) for item in error_lst]) + '\n')

                elif self.check_error:
                    # Print out the error metrics at each time step.
                    calc_error(self.config, self.model, self.gfu)

                elif self.save_error:
                    # Only saving the error metrics to file at each time step, so need to suppress the print statements
                    # from calc_error.
                    #with open(os.devnull, 'w') as f_tmp, contextlib.redirect_stdout(f_tmp):
                    error_lst = calc_error(self.config, self.model, self.gfu)
                    error_lst.insert(0, str(self.t_param[0].Get()))

                    # Write the calculated error metrics at the given time step to file.
                    with open(self.save_error_filename, 'a') as f:
                        f.write(', '.join([str(item) for item in error_lst]) + '\n')

        else:
            # Perform a stationary solve
            self._solve()

        # Save the final result
        if self.save_to_file:
            self.saver.save(self.gfu, self.t_param[0].Get())

            if self.model.DIM:
                self.saver.save(self.model.DIM_solver.phi_gfu, self.t_param[0].Get(), DIM=True)

        return self.gfu

    @abstractmethod
    def _startup(self) -> None:
        """
        Higher order methods need to be started up with a first order implicit solve.
        """

    @abstractmethod
    def _apply_boundary_conditions(self) -> None:
        """
        Apply the boundary conditions to all of the GridFunctions used by the model.
        """

    @abstractmethod
    def _assemble(self) -> None:
        """
        Assemble the linear and bilinear forms of the model.
        """

    @abstractmethod
    def _create_linear_and_bilinear_forms(self) -> None:
        """
        Create the linear and bilinear forms of the model and add any required time integration terms.
        """

    @abstractmethod
    def _create_preconditioners(self) -> None:
        """
        Create the preconditioner(s) used by this time integration scheme.
        """

    @abstractmethod
    def _update_preconditioners(self, precond_lst: List[Optional[Preconditioner]] = None) -> None:
        """
        Update the preconditioner(s) used by this time integration scheme. This is needed because the preconditioner(s)
        can't be updated if it is None.
        """

    @abstractmethod
    def _load_and_apply_initial_conditions(self) -> None:
        """
        Function to load the initial conditions.
        """

    @abstractmethod
    def _re_assemble(self) -> None:
        """
        Assemble the linear and bilinear forms of the model and update the preconditioner.
        """

    @abstractmethod
    def _single_solve(self) -> None:
        """
        Function to solve the model for the current time step.
        """

    @abstractmethod
    def _update_time_step(self) -> Tuple[bool, float, float, str]:
        """
        Function to calculate the new timestep and update the time (if a step is being taken).

        If the model is stationary, this does nothing.

        Returns:
            Tuple containing a bool, a float, and a string. The bool indicates whether or not the result of the current
            iteration was accepted based on all of the criteria established by the individual solver (a stationary
            solver MUST return True). The float indicates the current local error. The string contains the variable name
            for the variable with the highest local error.
        """
