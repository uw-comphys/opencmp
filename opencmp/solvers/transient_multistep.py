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
from .time_integration_schemes import implicit_euler, explicit_euler, euler_IMEX, crank_nicolson, CNLF, SBDF
from .base_solver import Solver
import ngsolve as ngs
from ngsolve import Preconditioner
from typing import List, Tuple

"""
Module for the multistep transient solver class.
"""


class TransientMultiStepSolver(Solver):
    """
    Transient multistep solver with a fixed time step.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

        if self.scheme_order > 1:
            # Need a startup.
            self._startup()

    def reset_model(self) -> None:
        super().reset_model()

        if self.scheme_order > 1:
            # Need a startup.
            self._startup()

    def _startup(self) -> None:
        # Use a very small time step and a fully implicit solve for the startup.
        self.dt_param[0].Set(1e-10)
        a, L = implicit_euler(self.model, self.gfu_0_list, self.dt_param)

        if self.model.preconditioners is not None:
            preconditioners = self.model.construct_preconditioners(a)

            for preconditioner in preconditioners:
                if preconditioner is not None:
                    preconditioner.Update()
        else:
            preconditioners = []

        while self.num_iters < self.scheme_order - 1:
            for i in range(len(self.t_param)):
                self.t_param[i].Set(self.t_param[i].Get() + self.dt_param[i].Get())

            self.model.apply_dirichlet_bcs_to(self.gfu)

            for i in range(len(a)):
                a[i].Assemble()
                L[i].Assemble()

            for preconditioner in preconditioners:
                if preconditioner is not None:
                    preconditioner.Update()

            self.model.solve_single_step(a, L, preconditioners, self.gfu)

            self.num_iters += 1

            # Update all previous timestep solutions.
            for i in range(1, self.scheme_order):
                self.gfu_0_list[-i].vec.data = self.gfu_0_list[-(i + 1)].vec

            # Update data in previous timestep with new solution
            self.gfu_0_list[0].vec.data = self.gfu.vec

            # Update the list of time step parameters.
            for i in range(1, self.scheme_order + 1):
                self.dt_param[-i].Set(self.dt_param[-(i + 1)].Get())

            # If another startup step must be taken the same very small dt value will be used.
            self.dt_param[0].Set(1e-10)

            # Update the model functions and boundary conditions with the new solution.
            self.model.update_model_variables(self.gfu_0_list[0])

            print('')
            print('Starting up.')
            print('---')

        # Now that the startup is over set the dt for the next time step to the original dt given in the config file.
        self.dt_param[0].Set(self.dt_param_init.Get())

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu)

    def _assemble(self) -> None:
        for i in range(len(self.a)):
            self.a[i].Assemble()
            self.L[i].Assemble()

        self._update_preconditioners()

    def _create_linear_and_bilinear_forms(self) -> None:
        if self.scheme == 'explicit euler':
            self.a, self.L = explicit_euler(self.model, self.gfu_0_list, self.dt_param)
        elif self.scheme == 'implicit euler':
            self.a, self.L = implicit_euler(self.model, self.gfu_0_list, self.dt_param)
        elif self.scheme == 'crank nicolson':
            self.a, self.L = crank_nicolson(self.model, self.gfu_0_list, self.dt_param)
        elif self.scheme == 'euler IMEX':
            self.a, self.L = euler_IMEX(self.model, self.gfu_0_list, self.dt_param)
        elif self.scheme == 'CNLF':
            self.a, self.L = CNLF(self.model, self.gfu_0_list, self.dt_param)
        elif self.scheme == 'SBDF':
            self.a, self.L = SBDF(self.model, self.gfu_0_list, self.dt_param)
        else:
            raise ValueError('Have not implemented {} time integration yet.'.format(self.scheme))

    def _create_preconditioners(self) -> None:
        self.preconditioners = self.model.construct_preconditioners(self.a)

    def _update_preconditioners(self, precond_lst: List[Optional[Preconditioner]] = None) -> None:
        for preconditioner in self.preconditioners:
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
        self.model.update_model_variables(self.gfu_0_list[0])

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        self.model.solve_single_step(self.a, self.L, self.preconditioners, self.gfu)

    def _update_time_step(self) -> Tuple[bool, float, float, str]:
        # Update all previous timestep solutions.
        for i in range(1, self.scheme_order):
            self.gfu_0_list[-i].vec.data = self.gfu_0_list[-(i + 1)].vec

        # Update data in previous timestep with new solution
        self.gfu_0_list[0].vec.data = self.gfu.vec

        # Update the list of time step parameters.
        for i in range(1, self.scheme_order + 1):
            self.dt_param[-i].Set(self.dt_param[-(i + 1)].Get())

        # Update the values of the model variables based on the previous timestep and re-parse the model functions as
        # necessary.
        self.model.update_model_variables(self.gfu_0_list[0])

        # Ensure we take a smaller step if we need to for reasons OTHER than local error.
        self.dt_param[0].Set(min(self.dt_param_init.Get(), self._dt_for_next_time_to_hit()))

        return True, -1.0, -1.0, ''
