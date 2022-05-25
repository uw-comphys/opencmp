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

from .base_controller import Controller
from ..models import Model
from ngsolve import CoefficientFunction, GridFunction, Parameter
from typing import Dict, List, Optional, cast
from numpy import isclose


class PID(Controller):
    """
    A Discrete PID controller.
    """

    def __init__(self, t_params: List[Parameter], model: Model, config_rel_path: str, import_dir: str) -> None:
        super().__init__(t_params, model, config_rel_path, import_dir)

        # Ensure that this is a Single-Input-Single-Output controller
        assert len(self.vars_control)    == len(t_params)
        assert len(self.vars_manipulate) == len(t_params)

        # Get information for manipulated variable
        self.bc_type     = self.vars_manipulate[0][0]
        self.bc_var      = self.vars_manipulate[0][1]
        self.bc_location = self.vars_manipulate[0][2]

        # Extract parameters
        parameters:  Dict[str, List[float]]   = cast(Dict[str, List[float]],
                                                     self.config_func.config.get_one_level_dict('PARAMETERS',
                                                                                                import_dir,
                                                                                                model.mesh,
                                                                                                [self.t_params[0]])[0])
        self.bias:   float              = parameters['bias'][0]
        self.K_c:    float              = parameters['k_c'][0]
        self.tau_I:  float              = parameters['tau_i'][0]
        self.tau_D:  float              = parameters['tau_d'][0]

        # List of past actions, needed for Integral and Derivative control
        # Initialize with initial error
        self.errors: List[float] = []

        # The integrated error used for the Integral portion of the controller
        self.integrated_error = 0.0

        # Load the initial BC value as the initial value to ensure that there's always a smooth transition
        # TODO: How do we handle it when the initial value is a coefficient function?
        initial_bc_val = self.model.BC.get(self.bc_type, {}).get(self.bc_var, {}).get(self.bc_location, None)[0]
        # TODO: when would this be none?
        if initial_bc_val is not None:
            if isinstance(initial_bc_val, float):
                self.prev_control_action_val = cast(float, initial_bc_val)
            else:
                raise ValueError('Only float BC types are currently supported by a PID controller')
        else:
            raise ValueError('Specified BC value does exist')

    def calculate_control_action(self, soln: GridFunction, rk_scheme: bool) \
            -> Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]]:

        control_action_dict: Dict[str, Dict[str, Dict[str, List[Optional[CoefficientFunction]]]]] = {}

        # DOES NOT get called for intermediate steps
        # BUT, needs to calculate a control action for each t_param so that each intermediate step has a value using
        # it's t_param so that it
        if isclose(self.t_next_action, self.model.t_param[0].Get()):
            # Get the measure value of the control variable
            # NOTE: [0] is used since a list is returned and for a PID that list has a single entry
            measurement = self._evaluate_control_variables(soln)[0]
            # Get the setpoint value for the control variable
            # NOTE: [0] is used vars_control is a list of control var info and for a PID that  list has a single entry
            # NOTE: [2] is used since the structure of self.vars_control[0] is [name_1, pos_1, val_1], and we want val_1
            setpoint = self.vars_control[0][2]
            # The index which to index into the setpoint and measurement values.
            # NOTE: For scalar trial functions this is 0 (-1 could also work)
            #       For a vector trial function (e.g. velocity) this will compare only a single component
            index = self.vars_control[0][3]

            # Calculate error
            error = setpoint[index] - measurement[index]

            # Store error at current timestep
            self.errors.append(error)

            # Set-point bias term
            control_action_val = self.bias
            # Proportional portion
            control_action_val += self.K_c * error
            # Integral portion
            try:
                control_action_val += self.K_c / self.tau_I * self._error_integral()
            except ZeroDivisionError:
                pass
            # Derivative portion
            control_action_val += self.K_c * self.tau_D * self._error_derivative()

            tmp_list: List[Optional[CoefficientFunction]] = [None for _ in range(len(self.t_params))]

            if rk_scheme:
                # TODO: Expand on below comments
                # Do not use the last entry for t_params for an RK scheme.
                for i in range(len(self.t_params)-1):
                    control_action = self._apply_dynamics_equation(control_action_val, self.prev_control_action_val,
                                                                   self.t_params[i])
                    tmp_list[i] = control_action

            else:
                # Rather than a perfect step change, the control action now follows a smoothed step change
                control_action = self._apply_dynamics_equation(control_action_val, self.prev_control_action_val,
                                                               self.t_params[0])

                tmp_list[0] = control_action

            # TODO: add to the correct location, merge correctly
            # TODO: for RK scheme, all BUT THE LAST positions get filled in
            # len(BC_list) == scheme_order + 1
            # DO NOT add values to BC_list[end]
            # if RK and self.scheme_order == 3 [ca_1, ca_2, ca_3, None] -> [t^n_1, t^int_2, t^int_1, t^n]
            # if not RK only modify BC_list[0]
            # Package control action into format needed to apply as BC
            control_action_dict = {self.bc_type: {self.bc_var: {self.bc_location: tmp_list}}}

            # Store this control action for later use
            self.prev_control_action_val = control_action_val

            # Increment the time for the next control action
            self._update_time_of_next_action()

        return control_action_dict

    def _error_integral(self) -> float:
        """
        Function to estimate the integral term for the controller
        """
        n = len(self.errors)

        if n >= 2:  # Trapezoid rule
            self.integrated_error += (self.errors[-1] + self.errors[-2]) / 2 * self.dt_control
        elif n == 1:  # Left hand rectangle rule
            self.integrated_error += self.errors[-1] * self.dt_control

        return self.integrated_error

    def _error_derivative(self) -> float:
        """
        Function to estimate the derivative term for the controller
        """
        derivative_error = 0.0

        # Number of previous errors stored
        n = len(self.errors)

        # TODO: implement higher order schemes once there is the data for it
        if n >= 3:  # 2nd order backwards difference
            derivative_error = (3 * self.errors[-1] - 4 * self.errors[-2] + self.errors[-3]) / self.dt_control
        elif n >= 2:  # 1st order backwards difference
            derivative_error = (self.errors[-1] - self.errors[-2]) / self.dt_control

        return derivative_error
