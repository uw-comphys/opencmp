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

"""
Tests for opencmp/post_processing/pyvista_visualizer.py.

pyvista, vtk, pyvistaqt, and qtpy (plus a Qt binding) are part of the `viz` extra and
are pulled in by `pip install -e .[all]`, same as tests.yml uses, so this file assumes
they are installed and imports them directly, no `pytest.importorskip` needed.

`QT_QPA_PLATFORM` is forced to `offscreen` before qtpy is imported so the Qt-backed
tests run headless, same idea as `LIBGL_ALWAYS_SOFTWARE` being needed for VTK's
software rendering fallback on a display-less runner.
"""

import os

if 'DISPLAY' not in os.environ and 'WAYLAND_DISPLAY' not in os.environ:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import platform
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import pyvista as pv
from pytest import CaptureFixture, MonkeyPatch, fixture
from qtpy import QtWidgets

from opencmp.config_functions import ConfigParser
from opencmp.post_processing import pyvista_visualizer as viz

_CONFIG_BLANK = str(Path(__file__).parent / 'config_blank')


def test_running_in_wsl(monkeypatch: MonkeyPatch) -> None:
    """
    Check that _running_in_wsl is False on a plain Linux box and True once
    WSL_DISTRO_NAME is set. All three detection signals are neutralized (env var,
    platform.uname().release, /proc/version) so this is hermetic even when actually
    run on a WSL host, where the latter two would otherwise already read "microsoft".
    """
    monkeypatch.delenv('WSL_DISTRO_NAME', raising=False)
    # _running_in_wsl only reads .release off the return value, so a SimpleNamespace
    # stand-in avoids relying on platform.uname_result's exact constructor signature
    # (which varies -- `processor` in particular is a lazily computed property, not a
    # constructor argument, on some Python versions).
    monkeypatch.setattr(platform, 'uname', lambda: SimpleNamespace(release='6.1.0-generic'))

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == '/proc/version':
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr('builtins.open', fake_open)

    assert viz._running_in_wsl() is False

    monkeypatch.setenv('WSL_DISTRO_NAME', 'Ubuntu')
    assert viz._running_in_wsl() is True


def _touch_vtu_files(tmp_path: Path, model_name: str, times: List) -> Path:
    """
    Build a `<tmp_path>/output/<model_name>_vtu/` directory containing one empty
    `.vtu` file per entry in `times`, named the way `sol_to_vtu` names them
    (`<model_name>_<time>.vtu`). `_read_time_series` only ever reads filenames, never
    file contents, so empty files are sufficient here.

    Args:
        tmp_path: The pytest tmp_path fixture root.
        model_name: model.name() equivalent.
        times: Values to build filenames from (str()'d directly).

    Returns:
        The output_dir (parent of the `<model_name>_vtu` directory) -- what gets
        passed as `_read_time_series`'s `output_dir` argument.
    """
    output_dir = tmp_path / 'output'
    vtu_dir = output_dir / f'{model_name}_vtu'
    vtu_dir.mkdir(parents=True)

    for t in times:
        (vtu_dir / f'{model_name}_{t}.vtu').touch()

    return output_dir


class TestReadTimeSeries:
    """ Class to test pyvista_visualizer._read_time_series. """

    def test_missing_vtu_dir_raises(self, tmp_path: Path) -> None:
        """ Check that an error is raised if the model's .vtu output directory doesn't exist. """
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            viz._read_time_series(output_dir, 'model')

    def test_empty_vtu_dir_raises(self, tmp_path: Path) -> None:
        """ Check that an error is raised if the .vtu directory exists but holds no .vtu files. """
        output_dir = tmp_path / 'output'
        (output_dir / 'model_vtu').mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            viz._read_time_series(output_dir, 'model')

    def test_numeric_sort_not_lexicographic(self, tmp_path: Path) -> None:
        """
        Check that the returned (time, path) pairs are sorted numerically, not as strings.
        Uses the exact adversarial example from the function's own docstring.
        """
        times = [9.0, 10.0, 1.953125e-08, 0.0002]
        output_dir = _touch_vtu_files(tmp_path, 'model', times)

        result = viz._read_time_series(output_dir, 'model')

        result_times = [t for t, _ in result]
        assert result_times == sorted(times)

    def test_single_file_stationary_case(self, tmp_path: Path) -> None:
        """ Check that a single saved file (a stationary run) is handled without special-casing. """
        output_dir = _touch_vtu_files(tmp_path, 'model', [0.0])

        result = viz._read_time_series(output_dir, 'model')

        assert len(result) == 1
        assert result[0][0] == 0.0

    def test_model_name_with_underscore_does_not_break_time_parsing(self, tmp_path: Path) -> None:
        """
        Check that a model name containing underscores doesn't confuse the time parsing, since
        parsing splits the filename stem on '_' and takes only the last field.
        """
        model_name = 'lid_driven_cavity'
        output_dir = _touch_vtu_files(tmp_path, model_name, [0.5])

        result = viz._read_time_series(output_dir, model_name)

        assert len(result) == 1
        assert result[0][0] == 0.5


@fixture
def small_pv_mesh() -> pv.UnstructuredGrid:
    """
    A small real pyvista mesh with a scalar field 'p' and a 2-component vector field
    'u' as point data, for _discover_fields/_update_vector_fields tests.

    Returns:
        A pv.UnstructuredGrid with 'p' (scalar) and 'u' (2-component vector) point data.
    """
    mesh = pv.Plane(i_resolution=3, j_resolution=3).cast_to_unstructured_grid()
    n = mesh.n_points
    mesh.point_data['p'] = np.linspace(0.0, 1.0, n)
    mesh.point_data['u'] = np.column_stack([np.linspace(0.0, 1.0, n), np.linspace(1.0, 0.0, n)])
    return mesh


class TestDiscoverFields:
    """ Class to test pyvista_visualizer._discover_fields. """

    def test_scalar_field_classified_as_not_vector(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that a 1D point data array is classified as a scalar field. """
        fields = viz._discover_fields(small_pv_mesh, ['p'], ['p', 'u'])

        assert fields == {'p': False}

    def test_2component_vector_classified_as_vector(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that a 2-component point data array is classified as a vector field. """
        fields = viz._discover_fields(small_pv_mesh, ['u'], ['p', 'u'])

        assert fields == {'u': True}

    def test_3component_vector_classified_as_vector(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that a 3-component point data array is also classified as a vector field. """
        n = small_pv_mesh.n_points
        small_pv_mesh.point_data['u3'] = np.column_stack(
            [np.linspace(0.0, 1.0, n), np.linspace(1.0, 0.0, n), np.zeros(n)])

        fields = viz._discover_fields(small_pv_mesh, ['u3'], ['p', 'u', 'u3'])

        assert fields == {'u3': True}

    def test_scalar_and_vector_together(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that requesting a mix of scalar and vector variables classifies each correctly. """
        fields = viz._discover_fields(small_pv_mesh, ['p', 'u'], ['p', 'u'])

        assert fields == {'p': False, 'u': True}

    def test_missing_variable_raises_and_lists_available(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that requesting a variable not in save_names raises, listing the available names. """
        with pytest.raises(ValueError) as excinfo:
            viz._discover_fields(small_pv_mesh, ['missing'], ['p', 'u'])

        assert 'p' in str(excinfo.value)
        assert 'u' in str(excinfo.value)


class TestUpdateVectorFields:
    """ Class to test pyvista_visualizer._update_vector_fields. """

    def test_scalar_fields_are_skipped(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that scalar fields don't get magnitude/3d arrays added. """
        fields = {'p': False}

        viz._update_vector_fields(small_pv_mesh, fields, {})

        assert 'p_magnitude' not in small_pv_mesh.point_data
        assert 'p_3d' not in small_pv_mesh.point_data

    def test_vector_magnitude_is_computed(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that a vector field's magnitude array matches np.linalg.norm of the raw data. """
        fields = {'u': True}
        expected = np.linalg.norm(small_pv_mesh.point_data['u'], axis=1)

        viz._update_vector_fields(small_pv_mesh, fields, {})

        assert np.allclose(small_pv_mesh.point_data['u_magnitude'], expected)

    def test_color_mode_does_not_add_3d_array(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that a vector field left at the default 'color' mode gets no '_3d' array. """
        fields = {'u': True}

        viz._update_vector_fields(small_pv_mesh, fields, {})

        assert 'u_3d' not in small_pv_mesh.point_data

    def test_lic_mode_pads_2component_vector_to_3d(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that lic mode appends a zero column to a 2-component vector field. """
        fields = {'u': True}
        original = small_pv_mesh.point_data['u'].copy()

        viz._update_vector_fields(small_pv_mesh, fields, {'u': 'lic'})

        result = small_pv_mesh.point_data['u_3d']
        assert result.shape == (small_pv_mesh.n_points, 3)
        assert np.allclose(result[:, :2], original)
        assert np.allclose(result[:, 2], 0.0)

    def test_lic_mode_leaves_3component_vector_unchanged(self, small_pv_mesh: pv.UnstructuredGrid) -> None:
        """ Check that lic mode leaves an already-3-component vector field's values unchanged. """
        n = small_pv_mesh.n_points
        small_pv_mesh.point_data['u3'] = np.column_stack(
            [np.linspace(0.0, 1.0, n), np.linspace(1.0, 0.0, n), np.full(n, 2.0)])
        fields = {'u3': True}

        viz._update_vector_fields(small_pv_mesh, fields, {'u3': 'lic'})

        assert np.allclose(small_pv_mesh.point_data['u3_3d'], small_pv_mesh.point_data['u3'])


@fixture
def pv_time_series(tmp_path: Path) -> SimpleNamespace:
    """
    Writes 3 small real .vtu frames under `<tmp_path>/output/<model_name>_vtu/`, each
    with a scalar field 'p' and a 2-component vector field 'u' that vary with time, for
    use by _build_plotter/visualize_results tests.

    Returns:
        A SimpleNamespace with:
            output_dir: The 'output' directory.
            model_name: The model name used in the .vtu filenames.
            files: The List[Tuple[float, Path]] _read_time_series returns for this dir.
            fields: {'p': False, 'u': True}, as _discover_fields would classify them.
    """
    model_name = 'model'
    times = [0.0, 0.5, 1.0]
    vtu_dir = tmp_path / 'output' / f'{model_name}_vtu'
    vtu_dir.mkdir(parents=True)

    for t in times:
        mesh = pv.Plane(i_resolution=3, j_resolution=3).cast_to_unstructured_grid()
        n = mesh.n_points
        mesh.point_data['p'] = np.linspace(0.0, 1.0, n) + t
        mesh.point_data['u'] = np.column_stack([np.full(n, t), np.linspace(0.0, 1.0, n)])
        mesh.save(vtu_dir / f'{model_name}_{t}.vtu')

    output_dir = tmp_path / 'output'
    files = viz._read_time_series(output_dir, model_name)

    return SimpleNamespace(output_dir=output_dir, model_name=model_name, files=files,
                           fields={'p': False, 'u': True})


@pytest.mark.slow
class TestBuildPlotter:
    """
    Class to test pyvista_visualizer._build_plotter. _build_plotter always builds the
    Qt-backed window now, so these tests depend on the session-scoped `qapp` fixture
    to ensure a QApplication exists. Marked slow since building a real VTK plotter per
    test is comparatively expensive; run with `pytest --runslow` to include this class.
    """

    def test_returns_plotter_with_one_row_per_variable(self, qapp: QtWidgets.QApplication,
                                                        pv_time_series: SimpleNamespace) -> None:
        """ Check that the returned plotter has one subplot row per requested variable. """
        mesh = pv.read(pv_time_series.files[0][1])

        plotter, render_frame, window = viz._build_plotter(
            mesh, pv_time_series.fields, {}, pv_time_series.files)

        assert window is not None
        assert plotter.shape == (len(pv_time_series.fields), 1)
        plotter.close()

    def test_vector_panels_render_after_scalar_panels(self, qapp: QtWidgets.QApplication,
                                                       pv_time_series: SimpleNamespace,
                                                       monkeypatch: MonkeyPatch) -> None:
        """
        Check that scalar variables always come before vector variables in the plotter's
        subplot rows, regardless of the order they were requested in -- see
        _build_plotter's `variables = sorted(fields, key=lambda var: fields[var])` line.
        Recorded via the order plotter.add_text labels are added, one per panel.
        """
        mesh = pv.read(pv_time_series.files[0][1])
        # Deliberately request the vector field before the scalar field.
        fields = {'u': True, 'p': False}

        labels_added = []
        from pyvistaqt import QtInteractor

        def make_recording(real_add_text):
            def recording_add_text(self, text, *args, **kwargs):
                labels_added.append(text)
                return real_add_text(self, text, *args, **kwargs)
            return recording_add_text

        for cls in (pv.Plotter, QtInteractor):
            monkeypatch.setattr(cls, 'add_text', make_recording(cls.add_text))
 

        plotter, render_frame, _ = viz._build_plotter(
            mesh, fields, {}, pv_time_series.files)

        assert labels_added == ['p', 'u']
        plotter.close()

    def test_render_frame_returns_correct_time(self, qapp: QtWidgets.QApplication,
                                               pv_time_series: SimpleNamespace) -> None:
        """ Check that render_frame(idx) returns the simulation time for that frame. """
        mesh = pv.read(pv_time_series.files[0][1])

        plotter, render_frame, _ = viz._build_plotter(
            mesh, pv_time_series.fields, {}, pv_time_series.files)

        for idx, (expected_time, _) in enumerate(pv_time_series.files):
            assert render_frame(idx) == expected_time
        plotter.close()

    def test_render_frame_does_not_reload_same_index(self, qapp: QtWidgets.QApplication,
                                                     pv_time_series: SimpleNamespace,
                                                     monkeypatch: MonkeyPatch) -> None:
        """ Check that calling render_frame twice with the same idx only reads the .vtu file once. """
        mesh = pv.read(pv_time_series.files[0][1])
        plotter, render_frame, _ = viz._build_plotter(
            mesh, pv_time_series.fields, {}, pv_time_series.files)

        read_calls = []
        real_read = pv.read

        def counting_read(path, *args, **kwargs):
            read_calls.append(path)
            return real_read(path, *args, **kwargs)

        monkeypatch.setattr(viz.pv, 'read', counting_read)

        render_frame(1)
        render_frame(1)
        assert len(read_calls) == 1

        render_frame(2)
        assert len(read_calls) == 2
        plotter.close()

    def test_lic_mode_builds_without_raising(self, qapp: QtWidgets.QApplication,
                                             pv_time_series: SimpleNamespace) -> None:
        """ Check that vector_plot_mode 'lic' exercises the vtkSurfaceLICMapper branch without error. """
        mesh = pv.read(pv_time_series.files[0][1])

        plotter, render_frame, _ = viz._build_plotter(
            mesh, pv_time_series.fields, {'u': 'lic'}, pv_time_series.files)

        for idx in range(len(pv_time_series.files)):
            render_frame(idx)
        plotter.close()

    def test_unrecognized_vector_plot_mode_raises(self, qapp: QtWidgets.QApplication,
                                                   pv_time_series: SimpleNamespace) -> None:
        """ Check that an unrecognized vector_plot_mode value raises a ValueError. """
        mesh = pv.read(pv_time_series.files[0][1])

        with pytest.raises(ValueError):
            viz._build_plotter(mesh, pv_time_series.fields, {'u': 'not_a_real_mode'},
                               pv_time_series.files)


class TestVisualizeResults:
    """ Class to test pyvista_visualizer.visualize_results. """

    def test_non_2d_model_skips_and_prints_message(self, capsys: CaptureFixture) -> None:
        """ Check that a non-2D model is skipped with a message, without ever touching config_parser. """
        model = SimpleNamespace(mesh=SimpleNamespace(dim=3))

        viz.visualize_results(None, model)

        captured = capsys.readouterr()
        assert '2D' in captured.out

    def test_empty_plot_variables_skips_and_prints_message(self, capsys: CaptureFixture) -> None:
        """ Check that an empty plot_variables list (the config default) is skipped with a message. """
        config_parser = ConfigParser(_CONFIG_BLANK)
        model = SimpleNamespace(mesh=SimpleNamespace(dim=2))

        viz.visualize_results(config_parser, model)

        captured = capsys.readouterr()
        assert 'plot_variables' in captured.out

    @pytest.mark.slow
    def test_full_run_completes(self, qapp: QtWidgets.QApplication, pv_time_series: SimpleNamespace,
                                monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(QtWidgets.QApplication, 'exec_', lambda self: None)

        config_parser = ConfigParser(_CONFIG_BLANK)
        config_parser['VISUALIZATION'] = {'plot_variables': 'p, u'}
        config_parser['OTHER'] = {'run_dir': str(pv_time_series.output_dir.parent)}

        model = SimpleNamespace(
            mesh=SimpleNamespace(dim=2),
            name=lambda: pv_time_series.model_name,
            save_names=['p', 'u'],
            run_dir=str(pv_time_series.output_dir.parent))

        viz.visualize_results(config_parser, model)


@fixture(scope='session')
def qapp() -> QtWidgets.QApplication:
    """
    Session-scoped QApplication for the Qt-dependent tests below. tests.yml runs
    pytest with --forked, so each test process gets its own, making session scope
    here safe.

    Returns:
        The (possibly pre-existing) QApplication instance.
    """
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@fixture
def fake_render_frame():
    """
    Stand-in for the render_frame closure _build_plotter would normally supply.
    Records every idx it's called with and returns float(idx) as a stand-in time.

    Returns:
        A callable (idx: int) -> float, with a `.calls` list attribute.
    """
    def render_frame(idx: int) -> float:
        render_frame.calls.append(idx)
        return float(idx)
    render_frame.calls = []
    return render_frame


class TestControlWindow:
    """ Class to test pyvista_visualizer._build_control_window / _ControlWindow. """

    N_FRAMES = 10

    def _make_window(self, qapp: QtWidgets.QApplication, fake_render_frame, n_frames: int = N_FRAMES):
        """
        Build a _ControlWindow and prime it exactly like production use eventually would.

        __init__ only *schedules* the first refresh() via QTimer.singleShot(0, ...),
        deferring it to the Qt event loop -- which never runs in these tests. Without
        that first refresh(), self._idx doesn't exist yet (only self.idx does; see
        __init__ vs refresh()), so step_forward/step_backward/tick all raise
        AttributeError if called first. Calling refresh(0) here stands in for that
        first event-loop tick.
        """
        window = viz._build_control_window(fake_render_frame, n_frames)
        window.refresh(0)
        fake_render_frame.calls.clear()
        return window

    def test_jump_to_start(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that jump_to_start resets to frame 0 and pauses playback. """
        window = self._make_window(qapp, fake_render_frame)
        window.scrub(5)

        window.jump_to_start()

        assert window._idx == 0
        assert window._playing is False
        assert window.btn_play.text() == 'Play'

    def test_jump_to_end(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that jump_to_end moves to the last frame and pauses playback. """
        window = self._make_window(qapp, fake_render_frame)

        window.jump_to_end()

        assert window._idx == self.N_FRAMES - 1
        assert window._playing is False

    def test_step_forward_respects_jump_size(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that step_forward advances by the configured jump size. """
        window = self._make_window(qapp, fake_render_frame)
        window.set_jump_size(3)

        window.step_forward()

        assert window._idx == 3

    def test_step_forward_clamps_at_last_frame(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that stepping forward past the last frame stays clamped at the last frame. """
        window = self._make_window(qapp, fake_render_frame)
        window.jump_to_end()

        window.step_forward()

        assert window._idx == self.N_FRAMES - 1

    def test_step_backward_respects_jump_size(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that step_backward retreats by the configured jump size. """
        window = self._make_window(qapp, fake_render_frame)
        window.set_jump_size(2)
        window.scrub(5)

        window.step_backward()

        assert window._idx == 3

    def test_step_backward_clamps_at_zero(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that stepping backward past frame 0 stays clamped at frame 0. """
        window = self._make_window(qapp, fake_render_frame)
        window.jump_to_start()

        window.step_backward()

        assert window._idx == 0

    def test_scrub_sets_index_and_pauses(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that scrubbing the slider jumps directly to the given index and pauses playback. """
        window = self._make_window(qapp, fake_render_frame)

        window.scrub(4)

        assert window._idx == 4
        assert window._playing is False

    def test_toggle_play_flips_state_and_label(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that toggle_play flips both the playing flag and the button label each call. """
        window = self._make_window(qapp, fake_render_frame)
        was_playing = window._playing

        window.toggle_play()
        assert window._playing is not was_playing
        assert window.btn_play.text() == ('Pause' if window._playing else 'Play')

        window.toggle_play()
        assert window._playing is was_playing

    def test_tick_noop_when_paused(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that tick does nothing while playback is paused. """
        window = self._make_window(qapp, fake_render_frame)
        window.toggle_play()  # pause (starts playing by default)
        idx_before = window._idx

        window.tick()

        assert window._idx == idx_before

    def test_tick_advances_when_playing(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that tick advances the frame index by the jump size while playing. """
        window = self._make_window(qapp, fake_render_frame)
        window.set_jump_size(1)
        assert window._playing is True

        window.tick()

        assert window._idx == 1

    def test_tick_autopauses_at_last_frame(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that reaching the last frame during playback auto-pauses instead of overshooting. """
        window = self._make_window(qapp, fake_render_frame)
        window.refresh(self.N_FRAMES - 1)
        window._playing = True

        window.tick()

        assert window._idx == self.N_FRAMES - 1
        assert window._playing is False
        assert window.btn_play.text() == 'Play'

    def test_refresh_calls_render_frame_once_and_updates_slider(self, qapp: QtWidgets.QApplication,
                                                                fake_render_frame) -> None:
        """ Check that refresh(idx) calls render_frame exactly once and syncs the slider position. """
        window = self._make_window(qapp, fake_render_frame)
        fake_render_frame.calls.clear()

        window.refresh(7)

        assert fake_render_frame.calls == [7]
        assert window.slider.value() == 7

    def test_single_frame_construction_does_not_crash(self, qapp: QtWidgets.QApplication, fake_render_frame) -> None:
        """ Check that a 1-frame (stationary) result set doesn't break the jump-size spin box's range. """
        window = self._make_window(qapp, fake_render_frame, n_frames=1)

        assert window.jump_size_box.minimum() == 1
        assert window.jump_size_box.maximum() == 1

    def test_close_event_quits_application(self, qapp: QtWidgets.QApplication, fake_render_frame,
                                          monkeypatch: MonkeyPatch) -> None:
        """ Check that closing the control window quits the QApplication. """
        window = self._make_window(qapp, fake_render_frame)

        quit_calls = []
        monkeypatch.setattr(QtWidgets.QApplication, 'quit', lambda self: quit_calls.append(True))

        window.close()

        assert quit_calls == [True]