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

import shutil
from pathlib import Path
from typing import List

import ngsolve as ngs
import pytest

from opencmp.config_functions import ConfigParser
from opencmp.models.misc import get_model_class
from opencmp.run import run
from opencmp.solvers.base_solver import scheme_history_order, scheme_order
from opencmp.solvers.misc import get_solver_class

# Keep these expectations independent of the production mapping so the test catches accidental confusion between
# previous-step history and Runge-Kutta stage count.
MULTISTEP_SCHEMES = ['CNLF', 'SBDF', 'adaptive IMEX']
RESTARTABLE_MULTISTAGE_SCHEMES = ['implicit euler', 'adaptive three step', 'RK 222', 'RK 232']

TEMPLATE = Path('pytests/full_system/restart/transient_poisson')
MESH = 'pytests/mesh_files/unit_square_coarse.vol'
INTERPOLANT_ORDER = 3

T_RESTART = 0.05
T_FINAL = 0.1


def _make_case(tmp_path: Path) -> Path:
    """ Copy the template run directory into tmp_path and point run_dir at the copy. """
    case = tmp_path / 'transient_poisson'
    shutil.copytree(TEMPLATE, case)

    config = case / 'config'
    config.write_text(config.read_text().replace('run_dir = .', 'run_dir = ' + str(case)))

    return case


def _sol_time(sol_file: Path) -> float:
    """ Recover the timestep a .sol file was saved at from its name. """
    return float(sol_file.stem.split('_')[1])


def _sol_files(sol_dir: Path) -> List[Path]:
    """
    All saved .sol files, oldest first. Selecting by parsed time rather than by formatted name is necessary because
    accumulated floating point error leaves names like "poisson_0.09999999999999999.sol".
    """
    return sorted(sol_dir.glob('poisson_*.sol'), key=_sol_time)


def _load(sol_path: Path, mesh: ngs.Mesh) -> ngs.GridFunction:
    """ Load a saved .sol file into a gridfunction on the model's finite element space. """
    gfu = ngs.GridFunction(ngs.H1(mesh, order=INTERPOLANT_ORDER))
    gfu.Load(str(sol_path))

    return gfu


@pytest.mark.parametrize('restart_selector', ['relative', 'absolute', 'LATEST', 'legacy'])
def test_resume_matches_uninterrupted_solve(tmp_path: Path, restart_selector: str, caplog) -> None:
    """
    Test that resuming a transient solve from a saved .sol file gives the same final solution as solving straight
    through. Fixed time-stepping (implicit euler) is used so that both solves take identical time steps and the
    resumed solve should reproduce the uninterrupted one to solver tolerance.
    """
    case = _make_case(tmp_path)
    sol_dir = case / 'output' / 'poisson_sol'

    # Solve 0.0 -> T_FINAL in one go and keep the final solution as the reference.
    run(str(case / 'config'))
    saved = _sol_files(sol_dir)
    assert _sol_time(saved[-1]) == pytest.approx(T_FINAL), 'The uninterrupted solve did not reach T_FINAL.'

    reference = tmp_path / 'reference_final.sol'
    shutil.copy(saved[-1], reference)

    # Simulate a crash at T_RESTART by throwing away every .sol saved after it.
    for sol_file in saved:
        if _sol_time(sol_file) > T_RESTART:
            sol_file.unlink()

    restart_from = _sol_files(sol_dir)[-1]
    assert _sol_time(restart_from) == pytest.approx(T_RESTART)

    # A similarly named checkpoint for another model must never be selected by LATEST.
    if restart_selector == 'LATEST':
        shutil.copy(restart_from, sol_dir / 'mcpoisson_999.0.sol')

    config = case / 'config'
    config_text = config.read_text().replace('resume_from_previous = False', 'resume_from_previous = True')
    if restart_selector == 'relative':
        config_text += '\nrestart_from = output/poisson_sol/' + restart_from.name + '\n'
    elif restart_selector == 'absolute':
        config_text += '\nrestart_from = ' + str(restart_from.resolve()) + '\n'
    elif restart_selector == 'LATEST':
        config_text += '\nrestart_from = LATEST\n'
    config.write_text(config_text)

    run(str(config))

    if restart_selector == 'legacy':
        assert 'without restart_from is deprecated' in caplog.text

    # Both solves should have ended up at the same place.
    resumed_final = _sol_files(sol_dir)[-1]
    assert _sol_time(resumed_final) == pytest.approx(T_FINAL), 'The resumed solve did not reach T_FINAL.'

    mesh = ngs.Mesh(MESH)
    expected = _load(reference, mesh)
    resumed = _load(resumed_final, mesh)

    err = ngs.sqrt(ngs.Integrate((expected - resumed) ** 2, mesh))
    norm = ngs.sqrt(ngs.Integrate(expected ** 2, mesh))

    assert norm > 0.0, 'Reference solution is identically zero, the test problem is not exercising anything.'
    assert err / norm < 1e-10


@pytest.mark.parametrize('scheme', MULTISTEP_SCHEMES)
def test_resume_rejects_multistep_schemes(tmp_path: Path, scheme: str) -> None:
    """
    Test that resuming is refused for schemes needing more than one previous time step. Only one saved solution is
    restored, and saved .sol files are not necessarily consecutive iterations, so the history a multi-step scheme
    needs cannot be reconstructed. Resuming anyway would silently re-run the scheme's startup and change the results.
    """
    case = _make_case(tmp_path)

    config = case / 'config'
    config.write_text(config.read_text()
                      .replace('scheme = implicit euler', 'scheme = ' + scheme)
                      .replace('resume_from_previous = False', 'resume_from_previous = True'))

    with pytest.raises(NotImplementedError, match='single-step'):
        run(str(config))


@pytest.mark.parametrize('scheme', RESTARTABLE_MULTISTAGE_SCHEMES)
def test_resume_allowed_for_single_step_schemes(tmp_path: Path, scheme: str) -> None:
    """ Test that restartable schemes pass the history guard, including multi-stage Runge-Kutta schemes. """
    case = _make_case(tmp_path)
    assert set(scheme_history_order) == set(scheme_order)
    assert scheme_history_order[scheme] == 1

    config = case / 'config'
    config.write_text(config.read_text()
                      .replace('scheme = implicit euler', 'scheme = ' + scheme)
                      .replace('resume_from_previous = False', 'resume_from_previous = True'))

    # Nothing has been solved yet so there is no .sol file to resume from. Constructing the solver exercises the resume
    # guard without requiring this Poisson fixture to implement the explicit IMEX terms used by the RK schemes.
    # Reaching the missing-checkpoint error means the scheme was accepted: a rejected scheme raises
    # NotImplementedError earlier, before any .sol file is looked for.
    config_parser = ConfigParser(str(config))
    model_class = get_model_class(config_parser.get_item(['OTHER', 'model'], str), False)

    with pytest.raises(FileNotFoundError, match='No valid .sol checkpoint'):
        get_solver_class(config_parser)(model_class, config_parser)


def test_resume_without_any_checkpoint_fails(tmp_path: Path) -> None:
    """
    Test that asking to resume when there is nothing to resume from is an error rather than a silent fresh start.
    A fresh start looks identical to a successful resume in the log, so it must not happen quietly.
    """
    case = _make_case(tmp_path)

    config = case / 'config'
    config.write_text(config.read_text().replace('resume_from_previous = False', 'resume_from_previous = True'))

    assert not (case / 'output').exists()

    with pytest.raises(FileNotFoundError, match='No valid .sol checkpoint'):
        run(str(config))


def test_explicit_restart_checkpoint_must_exist(tmp_path: Path) -> None:
    """ An explicit restart_from path must identify an existing file. """
    case = _make_case(tmp_path)
    config = case / 'config'
    config.write_text(
        config.read_text()
        .replace('resume_from_previous = False', 'resume_from_previous = True')
        + '\nrestart_from = output/poisson_sol/poisson_0.05.sol\n'
    )

    with pytest.raises(FileNotFoundError, match='does not exist'):
        run(str(config))


def test_restart_checkpoint_must_match_active_model(tmp_path: Path) -> None:
    """ An explicit checkpoint belonging to another model must be rejected before NGSolve loads it. """
    case = _make_case(tmp_path)
    sol_dir = case / 'output' / 'poisson_sol'
    run(str(case / 'config'))
    source = min(_sol_files(sol_dir), key=lambda path: abs(_sol_time(path) - T_RESTART))
    wrong_model = sol_dir / 'ins_0.05.sol'
    shutil.copy(source, wrong_model)

    config = case / 'config'
    config.write_text(
        config.read_text()
        .replace('resume_from_previous = False', 'resume_from_previous = True')
        + '\nrestart_from = ' + str(wrong_model) + '\n'
    )

    with pytest.raises(ValueError, match='active model'):
        run(str(config))


def test_restart_from_requires_resume_mode(tmp_path: Path) -> None:
    """ restart_from must not be silently ignored when resume_from_previous is disabled. """
    case = _make_case(tmp_path)
    config = case / 'config'
    config.write_text(config.read_text() + '\nrestart_from = LATEST\n')

    with pytest.raises(ValueError, match='resume_from_previous is False'):
        run(str(config))


def test_resume_of_completed_simulation_exits_cleanly(tmp_path: Path) -> None:
    """
    Test that resuming a simulation which already reached the end of time_range exits with a success code instead of
    taking a meaningless final step. The saved final time is short of the range end by floating point error, so this
    also covers the tolerance in the comparison.
    """
    case = _make_case(tmp_path)
    sol_dir = case / 'output' / 'poisson_sol'

    run(str(case / 'config'))

    final = _sol_files(sol_dir)[-1]
    assert _sol_time(final) != T_FINAL, 'Expected float drift in the saved time, the tolerance is untested without it.'
    assert _sol_time(final) == pytest.approx(T_FINAL)

    config = case / 'config'
    config.write_text(
        config.read_text().replace('resume_from_previous = False', 'resume_from_previous = True')
        + '\nrestart_from = LATEST\n'
    )

    with pytest.raises(SystemExit) as exit_info:
        run(str(config))

    assert exit_info.value.code == 0
