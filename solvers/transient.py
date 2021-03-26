"""
Module for the stationary solver class.
"""
from models import Model
from typing import Type
from config_functions import ConfigParser
from time_integration_schemes import implicit_euler, explicit_euler, crank_nicolson, CNLF, SBDF
from .base_solver import Solver
import ngsolve as ngs
from typing import List, Tuple


class TransientSolver(Solver):
    """
    Transient solver with a fixed time step.
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
        a, L = implicit_euler(self.model, self.gfu_0_list, self.U, self.V, ngs.Parameter(1e-10))
        preconditioner = self.model.construct_preconditioner(a)
        preconditioner.Update()

        while self.num_iters < self.scheme_order - 1:
            self.t_param.Set(self.t_param.Get() + 1e-10)

            self.model.apply_dirichlet_bcs_to(self.gfu)
            a.Assemble()
            L.Assemble()
            preconditioner.Update()
            self.model.single_iteration(a, L, preconditioner, self.gfu)

            self.num_iters += 1

            # Update all previous timestep solutions and dt values.
            for i in range(1, self.scheme_order):
                self.gfu_0_list[-i].vec.data = self.gfu_0_list[-(i + 1)].vec

            # Update data in previous timestep with new solution
            self.gfu_0_list[0].vec.data = self.gfu.vec

            print('')
            print('Starting up.')
            print('---')

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu)

    def _assemble(self) -> None:
        self.a.Assemble()
        self.L.Assemble()
        self.preconditioner.Update()

    def _create_linear_and_bilinear_forms(self) -> None:
        if self.scheme == 'explicit euler':
            self.a, self.L = explicit_euler(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        elif self.scheme == 'implicit euler':
            self.a, self.L = implicit_euler(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        elif self.scheme == 'crank nicolson':
            self.a, self.L = crank_nicolson(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        elif self.scheme == 'euler IMEX':
            self.a, self.L = implicit_euler(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        elif self.scheme == 'CNLF':
            self.a, self.L = CNLF(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        elif self.scheme == 'SBDF':
            self.a, self.L = SBDF(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        else:
            raise TypeError('Have not implemented {} time integration yet.'.format(self.scheme))

    def _create_preconditioner(self) -> None:
        self.preconditioner = self.model.construct_preconditioner(self.a)

    def _load_and_apply_initial_conditions(self) -> None:
        self.gfu_0_list: List[ngs.GridFunction] = []

        for i in range(self.scheme_order):
            gfu_0 = self.model.construct_gfu()
            gfu_0.vec.data = self.model.IC.vec
            self.gfu_0_list.append(gfu_0)

        # Update the values of the model variables based on the initial condition and re-parse the model functions as
        # necessary.
        self.model.update_model_variables(self.gfu_0_list[0])

    def _log_timestep(self, accepted: bool, error: float) -> None:
        if accepted:
            verb = 'Keeping'
        else:
            verb = 'Rejecting'

        print('')
        print('Current time: {}'.format(self.t_param.Get()))
        # Simple Transient solve returns a negative error to match function signature, but it is meaningless
        if error > 0:
            print('{} solve, local error = {}'.format(verb, error))
        print('New dt:       {}'.format(self.dt_param[0].Get()))
        print('---')

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        self.model.single_iteration(self.a, self.L, self.preconditioner, self.gfu)

    def _update_time_step(self) -> Tuple[bool, float]:
        # Update all previous timestep solutions and dt values.
        for i in range(1, self.scheme_order):
            self.gfu_0_list[-i].vec.data = self.gfu_0_list[-(i + 1)].vec
            self.dt_param[-i].Set(self.dt_param[-(i + 1)].Get())

        # Update data in previous timestep with new solution
        self.gfu_0_list[0].vec.data = self.gfu.vec

        # Update the values of the model variables based on the previous timestep and re-parse the model functions as
        # necessary.
        self.model.update_model_variables(self.gfu_0_list[0])

        # Ensure we take a smaller step if we need to for reasons OTHER than local error.
        self.dt_param[0].Set(min(self.dt_param_init.Get(), self._dt_for_next_time_to_hit()))

        return True, -1.0
