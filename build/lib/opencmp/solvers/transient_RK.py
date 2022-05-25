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

from ..models import Model
from typing import Type, Optional
from ..config_functions import ConfigParser
from .time_integration_schemes import RK_222, RK_232
from .base_solver import Solver
import ngsolve as ngs
from ngsolve import Preconditioner
from typing import List, Tuple

"""
Module for the Runge Kutta transient solver class.
"""


class TransientRKSolver(Solver):
    """
    Transient Runge Kutta solver with a fixed time step.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

        self.step = 1

    def reset_model(self) -> None:
        super().reset_model()

    def _startup(self) -> None:
        # Runge Kutta schemes are single-step schemes so should never need a startup.
        pass

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu, self.scheme_order - self.step)

    def _assemble(self) -> None:
        for i in range(len(self.a_list[-self.step])):
            self.a_list[-self.step][i].Assemble()
            self.L_list[-self.step][i].Assemble()
        self._update_preconditioners(self.preconditioner_list[-self.step])

    def _create_linear_and_bilinear_forms(self) -> None:
        # Each intermediate step has its own bilinear and linear form. Add the weak forms in reverse step order to be
        # consistent with the order of t_param, dt_param, and gfu_0_list.
        self.a_list = []
        self.L_list = []
        for i in range(self.scheme_order, 0, -1):
            if self.scheme == 'RK 222':
                a, L = RK_222(self.model, self.gfu_0_list, self.dt_param, i)
                self.a_list.append(a)
                self.L_list.append(L)
            elif self.scheme == 'RK 232':
                a, L = RK_232(self.model, self.gfu_0_list, self.dt_param, i)
                self.a_list.append(a)
                self.L_list.append(L)
            else:
                raise ValueError('Have not implemented {} time integration yet.'.format(self.scheme))

    def _create_preconditioners(self) -> None:
        # Each intermediate step needs its own preconditioner. Add the preconditioners in reverse step order to be
        # consistent with the order of t_param, dt_param, and gfu_0_list.
        self.preconditioner_list = [self.model.construct_preconditioners(a) for a in self.a_list]

    def _update_preconditioners(self, precond_lst: List[Optional[Preconditioner]] = None) -> None:
        for preconditioner in precond_lst:
            if preconditioner is not None:
                preconditioner.Update()

    def _load_and_apply_initial_conditions(self) -> None:
        self.gfu_0_list: List[ngs.GridFunction] = []

        for i in range(self.scheme_order):
            gfu_0 = self.model.construct_gfu()
            gfu_0.vec.data = self.model.IC.vec
            self.gfu_0_list.append(gfu_0)

        # Update the values of the model variables based on the initial condition and re-parse the model functions as
        # necessary.
        self.model.update_model_variables(self.gfu_0_list[0], time_step=self.scheme_order)

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        # Have already assembled the weak form and set the boundary conditions for the first intermediate step.
        # Solve the first intermediate step.
        self.model.solve_single_step(self.a_list[-1], self.L_list[-1], self.preconditioner_list[-1], self.gfu, self.scheme_order - 1)

        # Update self.gfu_0_list and self.step.
        self._update_intermediate_step()

        while self.step < self.scheme_order:
            # Apply boundary conditions for the next intermediate step.
            self._apply_boundary_conditions()

            # Assemble the weak form for the next intermediate step.
            self._re_assemble()

            # Solve the next intermediate step.
            self.model.solve_single_step(self.a_list[-self.step], self.L_list[-self.step], self.preconditioner_list[-self.step], self.gfu, self.scheme_order - self.step)

            # Update self.gfu_0_list and self.step.
            self._update_intermediate_step()

        # The intermediate steps are done, now solve for the actual time step solution.
        # Apply boundary conditions.
        self._apply_boundary_conditions()

        # Assemble the weak form.
        self._re_assemble()

        # Solve the time step.
        self.model.solve_single_step(self.a_list[0], self.L_list[0], self.preconditioner_list[0], self.gfu, 0)

    def _update_time_step(self) -> Tuple[bool, float, float, str]:
        # Set all intermediate step solutions to the current solution. Set all dt values to the next dt (may vary to
        # hit a save point).
        next_dt = min(self.dt_param_init.Get(), self._dt_for_next_time_to_hit())
        self.dt_param[-1].Set(self.dt_param[0].Get())
        for i in range(self.scheme_order):
            self.gfu_0_list[i].vec.data = self.gfu.vec
            self.dt_param[i].Set(next_dt * self.scheme_dt_coef[i])

        # Update the values of the model variables based on the previous timestep and re-parse the model functions as
        # necessary.
        self.model.update_model_variables(self.gfu, time_step=self.scheme_order)

        # Update the model component values for all the intermediate time steps to be the trial functions.
        for i in range(1, self.scheme_order):
            self.model.update_model_variables(self.U, time_step=i)

        # Reset self.step to one.
        self.step = 1

        return True, -1.0, -1.0, ''

    def _update_intermediate_step(self) -> None:
        # Set the appropriate intermediate step solution to the current solution.
        self.gfu_0_list[-(self.step + 1)].vec.data = self.gfu.vec

        # Update the values of the model variables for this intermediate step and re-parse the model functions as
        # necessary.
        self.model.update_model_variables(self.gfu, time_step=self.scheme_order - self.step + 1)

        # Update self.step.
        self.step += 1

