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

from opencmp.run import run

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


def test_resume_matches_uninterrupted_solve(tmp_path: Path) -> None:
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

    # Resume. The initial condition has to be pointed at the last .sol by hand, resume_from_previous only picks up
    # the new start time and checks that the initial condition agrees with the .sol file it found.
    (case / 'ic_dir' / 'ic_config').write_text(
        '[POISSON]\nu = all -> output/poisson_sol/' + restart_from.name + '\n'
    )
    config = case / 'config'
    config.write_text(config.read_text().replace('resume_from_previous = False', 'resume_from_previous = True'))

    run(str(config))

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
