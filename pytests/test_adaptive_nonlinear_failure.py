"""Tests for how the adaptive solvers react to a diverged nonlinear solve."""

from opencmp.solvers.adaptive_transient_solvers.adaptive_three_step import AdaptiveThreeStep
from opencmp.solvers.adaptive_transient_solvers.adaptive_two_step import AdaptiveTwoStep


class _Parameter:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value


class _Vector:
    def __init__(self, value):
        self.data = value


class _GridFunction:
    def __init__(self, value):
        self.vec = _Vector(value)


class _FailingModel:
    """Reports divergence on every solve."""

    def __init__(self):
        self.solve_calls = 0
        self.linearization = None

    def solve_single_step(self, *args, **kwargs):
        self.solve_calls += 1
        return False

    def update_linearization(self, gfu):
        self.linearization = gfu

    def update_model_variables(self, gfu, time_step=None):
        pass


def _two_step():
    solver = AdaptiveTwoStep.__new__(AdaptiveTwoStep)
    solver.model = _FailingModel()
    solver.a_pred = solver.L_pred = solver.preconditioner_pred = []
    solver.a_corr = solver.L_corr = solver.preconditioner_corr = []
    solver.gfu_0_list = [_GridFunction('accepted')]
    solver.gfu_pred = _GridFunction('failed predictor')
    solver.gfu = _GridFunction('failed corrector')
    return solver


def _three_step():
    solver = AdaptiveThreeStep.__new__(AdaptiveThreeStep)
    solver.model = _FailingModel()
    solver.a_long = solver.L_long = solver.preconditioner_long = []
    solver.scheme = 'adaptive three step'
    solver.scheme_order = 2
    solver.scheme_dt_coef = [1.0, 0.5]
    solver.gfu_0_list = [_GridFunction('intermediate'), _GridFunction('accepted')]
    solver.gfu_long = _GridFunction('failed long')
    solver.gfu_short = _GridFunction('failed short')
    solver.gfu = _GridFunction('failed final')
    return solver


def test_two_step_stops_and_discards_after_first_failure() -> None:
    solver = _two_step()
    accepted = solver.gfu_0_list[0]

    assert solver._single_solve() is False
    assert solver.model.solve_calls == 1
    assert solver.gfu_pred.vec.data is accepted.vec
    assert solver.gfu.vec.data is accepted.vec
    assert solver.model.linearization is accepted


def test_three_step_stops_and_discards_after_first_failure() -> None:
    solver = _three_step()
    accepted = solver.gfu_0_list[-1]

    assert solver._single_solve() is False
    assert solver.model.solve_calls == 1
    assert all(candidate.vec.data is accepted.vec for candidate in
               (solver.gfu_long, solver.gfu_short, solver.gfu))
    assert solver.model.linearization is accepted


def test_two_step_nonlinear_failure_rejects_and_shrinks_dt() -> None:
    solver = _two_step()
    solver.dt_range = [1e-8, 1.0]
    solver.dt_abs_tol = solver.dt_rel_tol = 1e-4
    solver.dt_param = [_Parameter(0.4), _Parameter(0.4)]
    solver.t_param = [_Parameter(1.0), _Parameter(0.6)]
    solver._dt_for_next_time_to_hit = lambda: 1.0

    accepted, _, _, component = solver._update_time_step(nonlinear_failed=True)

    assert accepted is False
    assert component == 'nonlinear solve'
    assert solver.dt_param[0].Get() < 0.4
    assert solver.t_param[0].Get() == 0.6


def test_three_step_nonlinear_failure_rejects_and_halves_dt() -> None:
    solver = _three_step()
    solver.dt_range = [1e-8, 1.0]
    solver.dt_abs_tol = solver.dt_rel_tol = 1e-4
    solver.dt_param = [_Parameter(0.4), _Parameter(0.2), _Parameter(0.4)]
    solver.t_param = [_Parameter(1.0), _Parameter(1.0), _Parameter(0.6)]
    solver.step = 2
    solver._dt_for_next_time_to_hit = lambda: 1.0

    accepted, _, _, component = solver._update_time_step(nonlinear_failed=True)

    assert accepted is False
    assert component == 'nonlinear solve'
    assert solver.dt_param[0].Get() == 0.2
    assert solver.dt_param[1].Get() == 0.1
    assert solver.t_param[0].Get() == 0.6
    assert solver.step == 1
