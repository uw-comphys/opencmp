import ngsolve as ngs
from models import Model
from typing import Tuple, Type, List, Union
from config_functions import ConfigParser
from time_integration_schemes import implicit_euler, crank_nicolson
from ..adaptive_transient import AdaptiveTransientSolver


class AdaptiveTwoStep(AdaptiveTransientSolver):
    """
    An adaptive time-stepping solver that uses a two-step method.

    Each timestep is solved first with implicit Euler (first-order) and then with Crank-Nicolson (second-order). The
    local error is estimated from the L2 norm of the difference between the solutions. If the timestep is accepted the
    implicit Euler solution is kept since implicit Euler is unconditionally stable.
    """

    def __init__(self, model_class: Type[Model], config: ConfigParser) -> None:
        super().__init__(model_class, config)

        self.gfu_pred = self.model.construct_gfu()

    def reset_model(self) -> None:
        super().reset_model()

        self.gfu_pred = self.model.construct_gfu()

    def _apply_boundary_conditions(self) -> None:
        self.model.apply_dirichlet_bcs_to(self.gfu_pred)
        self.model.apply_dirichlet_bcs_to(self.gfu)

    def _assemble(self) -> None:
        self.a_pred.Assemble()
        self.L_pred.Assemble()
        self.preconditioner_pred.Update()

        self.a_corr.Assemble()
        self.L_corr.Assemble()
        self.preconditioner_corr.Update()

    def _create_linear_and_bilinear_forms(self) -> None:
        self.a_pred, self.L_pred = crank_nicolson(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)
        self.a_corr, self.L_corr = implicit_euler(self.model, self.gfu_0_list, self.U, self.V, self.dt_param)

    def _create_preconditioner(self) -> None:
        self.preconditioner_pred = self.model.construct_preconditioner(self.a_pred)
        self.preconditioner_corr = self.model.construct_preconditioner(self.a_corr)

    def _re_assemble(self) -> None:
        self._assemble()

    def _single_solve(self) -> None:
        self.model.single_iteration(self.a_pred, self.L_pred, self.preconditioner_pred, self.gfu_pred)
        self.model.single_iteration(self.a_corr, self.L_corr, self.preconditioner_corr, self.gfu)

    def _calculate_local_error(self) -> Tuple[Union[str, List], List]:
        if len(self.gfu.components) == 0:
            # Only one model variable to estimate local error with.
            local_errors = [ngs.sqrt(ngs.Integrate((self.gfu - self.gfu_pred) ** 2, self.model.mesh))]

            # Also get the gridfunction norm to use for the relative error tolerance.
            gfu_norms = [ngs.sqrt(ngs.Integrate(self.gfu ** 2, self.model.mesh))]

        else:
            # Include any variables specified by the model as included in local error.
            local_errors = []

            # Also get the gridfunction norms to use for the relative error tolerance.
            gfu_norms = []

            for var, use in self.model.model_local_error_components.items():
                if use:
                    component = self.model.model_components[var]
                    local_errors.append(ngs.sqrt(ngs.Integrate((self.gfu.components[component] -
                                                                    self.gfu_pred.components[component]) ** 2,
                                                                    self.model.mesh)))
                    gfu_norms.append(ngs.sqrt(ngs.Integrate(self.gfu.components[component] ** 2, self.model.mesh)))

        return local_errors, gfu_norms
