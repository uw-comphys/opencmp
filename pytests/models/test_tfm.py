from pathlib import Path
import math
import re

import ngsolve as ngs
import pytest
from netgen.geom2d import unit_square

from opencmp.config_functions import ConfigParser
from opencmp.config_functions.boundary_conditions import BCFunctions
from opencmp.models.tfm import TwoFluidModel
import opencmp.models.tfm as tfm_module


def _injection_test_model() -> TwoFluidModel:
    model = TwoFluidModel.__new__(TwoFluidModel)
    model.mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.4))
    model.fes = ngs.FESpace([
        ngs.HDiv(model.mesh, order=1, dgjumps=True),
        ngs.HDiv(model.mesh, order=1, dgjumps=True),
        ngs.L2(model.mesh, order=0, dgjumps=True),
        ngs.L2(model.mesh, order=0, dgjumps=True),
    ], dgjumps=True)
    model.model_components = {'u_c': 0, 'u_d': 1, 'p': 2, 'alpha_c': 3}
    model.UIter = ngs.GridFunction(model.fes)
    model.UIter.components[3].Set(0.8)
    model.n = ngs.specialcf.normal(model.mesh.dim)
    model.IM = ngs.Id(model.mesh.dim)
    model.penalty_interior = 10.0
    model.rho_c = [1000.0]
    model.rho_d = [1.0]
    model.nu_c = [0.0]
    model.nu_d = [0.0]
    model.dp = [0.003]
    model.BC = {}
    model.dirichlet_names = {}
    model.f = {}
    model.diffusion_switch = False
    model.canonical_form = 'B-TFM'
    model.gravity = ngs.CoefficientFunction((0.0, 0.0))
    model.drag_switch = False
    model.VM_switch = False
    model.Disp_switch = False
    model.Lift_switch = False
    model.mean_zero_pressure = False
    model.injection_switch = True
    model.inj_mass_flowrate = 0.3
    model.injection_region = '.*'
    model.inj_velocity = 2.0
    return model


def _boundary_trace_integral(mesh, expression) -> float:
    """Integrate the adjacent volume trace of an L2 field on the boundary."""
    space = ngs.L2(mesh, order=0)
    test = space.TestFunction()
    one = ngs.GridFunction(space)
    one.Set(1.0)
    form = ngs.LinearForm(space)
    form += expression * test * ngs.ds(skeleton=True, bonus_intorder=10)
    form.Assemble()
    return ngs.InnerProduct(form.vec, one.vec)


def _tfm_from_config(tmp_path: Path, tfm: dict[str, str], other=None) -> TwoFluidModel:
    config_path = tmp_path / 'config'
    config_path.write_text('')
    config = ConfigParser(str(config_path))
    config['TFM'] = tfm
    config['OTHER'] = other or {}
    model = TwoFluidModel.__new__(TwoFluidModel)
    model.config = config
    model._pre_init()
    return model


_ALL_IME = ('drag -> Tomiyama\n'
            'lift -> LegendreMagnaudet\n'
            'virtual_mass -> ConstantCoefficient\n'
            'laminar_dispersion -> ConstantCoefficient')


def test_tfm_ime_enables_requested_mechanisms(tmp_path: Path) -> None:
    model = _tfm_from_config(tmp_path, {
        'canonical_form': 'C-TFM',
        'IME': _ALL_IME,
        'lift_wall_deactivation': 'True',
        'lift_wall_boundaries': 'wall|bottom',
    })

    assert model.canonical_form == 'C-TFM'
    assert model.drag_switch
    assert model.VM_switch
    assert model.Disp_switch
    assert model.Lift_switch
    assert model.drag_model == 'Tomiyama'
    assert model.lift_model == 'LegendreMagnaudet'
    assert model.lift_wall_deactivation
    assert model.lift_wall_boundaries == 'wall|bottom'


@pytest.mark.parametrize(('tfm', 'message'), [
    ({'IME': 'drag -> Tomiyama\nbuoyancy -> Tomiyama'}, 'Unknown [TFM] IME mechanism'),
    ({'IME': 'laminar_dispersion -> ConstantCoefficient'},
     "'laminar_dispersion' requires 'drag'"),
    ({'canonical_form': 'invalid'}, 'canonical_form'),
    ({'IME': 'drag -> invalid'}, "IME 'drag' must use one of"),
    ({'IME': 'lift -> invalid'}, "IME 'lift' must use one of"),
    ({'IME': 'virtual_mass -> Tomiyama'}, "IME 'virtual_mass' must use one of"),
    ({'drag_model': 'Tomiyama'}, 'Unknown [TFM] option'),
    ({'unexpected': 'value'}, 'Unknown [TFM] option'),
    ({'IME': 'drag -> Tomiyama', 'lift_wall_deactivation': 'True'},
     'lift_wall_deactivation requires lift'),
])
def test_tfm_configuration_rejects_invalid_values(
        tmp_path: Path, tfm: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _tfm_from_config(tmp_path, tfm)


def test_slip_accepts_marker_only_syntax(tmp_path: Path) -> None:
    bc_path = tmp_path / 'bc_config'
    bc_path.write_text('[SLIP]\nu_d = wall|bottom\n')

    functions = BCFunctions(str(bc_path), str(tmp_path), None,
                            ['dirichlet', 'neumann', 'slip'])
    boundary_conditions, dirichlet_names = functions.set_boundary_conditions(
        ['dirichlet', 'neumann', 'slip'])

    assert boundary_conditions['slip']['u_d'] == {'wall': [], 'bottom': []}
    assert 'u_d' not in dirichlet_names


def test_hdiv_space_combines_dirichlet_and_slip_markers(monkeypatch) -> None:
    model = TwoFluidModel.__new__(TwoFluidModel)
    model.element = {'u_c': 'HDiv', 'u_d': 'HDiv', 'p': 'L2', 'alpha_c': 'L2'}
    model.interp_ord = 2
    model.mesh = object()
    model.DG = True
    model.BC = {
        'dirichlet': {'u_c': {'inlet': []}, 'u_d': {'outlet': []}},
        'slip': {'u_c': {}, 'u_d': {'wall': [], 'bottom': []}},
    }

    hdiv_calls = []

    def fake_hdiv(mesh, **kwargs):
        hdiv_calls.append(kwargs)
        return object()

    monkeypatch.setattr(tfm_module.ngs, 'HDiv', fake_hdiv)
    monkeypatch.setattr(tfm_module.ngs, 'L2', lambda mesh, **kwargs: object())
    monkeypatch.setattr(tfm_module, 'FESpace', lambda spaces, **kwargs: spaces)

    model._construct_fes()

    assert hdiv_calls[0]['dirichlet'] == 'inlet'
    assert hdiv_calls[1]['dirichlet'] == 'outlet|wall|bottom'


def test_dirichlet_and_slip_cannot_overlap() -> None:
    model = TwoFluidModel.__new__(TwoFluidModel)
    model.BC = {
        'dirichlet': {'u_d': {'wall': []}},
        'slip': {'u_d': {'wall': []}},
    }

    with pytest.raises(ValueError, match='both DIRICHLET and SLIP'):
        model._velocity_space_boundaries('u_d')


def test_hdiv_slip_constrains_normal_but_preserves_tangential_velocity() -> None:
    mesh = ngs.Mesh('pytests/mesh_files/unit_square_coarse.vol')
    for _ in range(2):
        mesh.Refine()

    # In HDiv, marking the bottom as Dirichlet constrains only u.n. Project a
    # horizontal field with a nonzero tangential component toward that wall.
    fes = ngs.HDiv(mesh, order=2, dirichlet='bottom')
    u, v = fes.TnT()
    a = ngs.BilinearForm(fes)
    a += (u * v + 1e-3 * ngs.div(u) * ngs.div(v)) * ngs.dx
    L = ngs.LinearForm(fes)
    target = ngs.CoefficientFunction((1.0, 0.0))
    L += target * v * ngs.dx
    a.Assemble()
    L.Assemble()

    velocity = ngs.GridFunction(fes)
    velocity.vec.data = a.mat.Inverse(
        freedofs=fes.FreeDofs(), inverse='umfpack') * L.vec

    normal = ngs.specialcf.normal(mesh.dim)
    normal_l2_squared = ngs.Integrate(
        (velocity * normal)**2, mesh, ngs.BND,
        definedon=mesh.Boundaries('bottom'))
    near_wall_tangential_energy = ngs.Integrate(
        ngs.IfPos(0.125 - ngs.y, 1.0, 0.0) * velocity[0]**2, mesh)
    projection_error = ngs.Integrate((velocity - target)**2, mesh)

    print('TFM SLIP bottom normal L2^2: {:.6e}'.format(normal_l2_squared))
    print('TFM SLIP near-wall tangential energy: {:.6e}'.format(
        near_wall_tangential_energy))
    assert normal_l2_squared == pytest.approx(0.0, abs=1e-14)
    assert near_wall_tangential_energy > 0.1
    assert math.sqrt(projection_error) < 1e-12


def test_ime_closures_are_finite_and_zero_without_a_driver() -> None:
    config = ConfigParser('pytests/full_system/tfm/config')
    config.set('TFM', 'IME', _ALL_IME)
    config.set('TFM', 'lift_wall_deactivation', 'False')
    model = TwoFluidModel(config, [ngs.Parameter(0.0)])

    velocity = ngs.CoefficientFunction((1.0, 0.0))
    for alpha_c in (0.1, 0.5, 0.9):
        model.UIter.components[model.model_components['u_c']].Set(velocity)
        model.UIter.components[model.model_components['u_d']].Set(velocity)
        model.UIter.components[model.model_components['alpha_c']].Set(alpha_c)

        wc = model.UIter.components[model.model_components['u_c']]
        wd = model.UIter.components[model.model_components['u_d']]
        ac = model.UIter.components[model.model_components['alpha_c']]
        ad = 1 - ac
        cd = model._get_drag_coeff(wd, wc, ad, 0)
        cl = model._get_lift_coeff(wd, wc, 0)

        # With no slip, drag and dispersion have zero magnitude; lift also has
        # zero slip and zero carrier vorticity. Virtual mass is covered by its
        # MMS convergence and action-reaction regression tests.
        drag = cd * ngs.Norm(wd - wc) * (wd - wc)
        dispersion = cd * ngs.Norm(wd - wc)**2 * ngs.grad(ac)
        curl_wc = ngs.grad(wc)[1] - ngs.grad(wc)[2]
        lift = cl * curl_wc * ngs.CoefficientFunction(
            (wd[1] - wc[1], -(wd[0] - wc[0])))

        for closure in (drag, dispersion, lift):
            magnitude_squared = ngs.Integrate(closure * closure, model.mesh)
            assert math.isfinite(magnitude_squared)
            assert magnitude_squared == pytest.approx(0.0, abs=1e-24)


def test_injection_sources_use_consistent_equation_scaling_and_phase() -> None:
    model = _injection_test_model()
    dt = ngs.Parameter(2e-3)
    U, V = model.fes.TrialFunction(), model.fes.TestFunction()

    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear_time_ODE(U, V, dt, 0)[0]
    L = ngs.LinearForm(model.fes)
    L += model.construct_linear(V, None, dt, 0)[0]
    a.Assemble()
    L.Assemble()

    probes = []
    for component in range(4):
        probe = ngs.GridFunction(model.fes)
        if component < 2:
            probe.components[component].Set(ngs.CoefficientFunction((0.0, 1.0)))
        else:
            probe.components[component].Set(1.0)
        probes.append(probe)

    continuous_momentum = ngs.InnerProduct(L.vec, probes[0].vec)
    dispersed_momentum = ngs.InnerProduct(L.vec, probes[1].vec)
    mixture_mass = ngs.InnerProduct(L.vec, probes[2].vec)
    dispersed_mass = ngs.InnerProduct(L.vec, probes[3].vec)

    assert continuous_momentum == pytest.approx(0.0, abs=1e-14)
    assert dispersed_momentum == pytest.approx(6e-3)
    assert mixture_mass == pytest.approx(0.3)
    assert dispersed_mass == pytest.approx(-6e-4)

    injected_velocity = ngs.GridFunction(model.fes)
    injected_velocity.components[1].Set(ngs.CoefficientFunction((0.0, 2.0)))
    dilution = ngs.InnerProduct(a.mat * injected_velocity.vec, probes[1].vec)
    assert dilution == pytest.approx(dispersed_momentum)


def test_mixture_equation_includes_discontinuous_volume_fraction_flux() -> None:
    model = _injection_test_model()
    model.injection_switch = False
    model.UIter.components[3].Set(ngs.IfPos(ngs.x - 0.5, 0.8, 0.2))
    U, V = model.fes.TrialFunction(), model.fes.TestFunction()
    a = ngs.BilinearForm(model.fes)
    a += model.construct_bilinear_time_ODE(U, V, ngs.Parameter(2e-3), 0)[0]
    a.Assemble()

    state = ngs.GridFunction(model.fes)
    state.components[0].Set(ngs.CoefficientFunction((1.0, 0.0)))
    constant_q = ngs.GridFunction(model.fes)
    constant_q.components[2].Set(1.0)

    weak_mixture_flux = ngs.InnerProduct(a.mat * state.vec, constant_q.vec)
    physical_boundary_flux = _boundary_trace_integral(
        model.mesh, model.UIter.components[3] * state.components[0] * model.n)

    assert physical_boundary_flux > 0.1
    assert weak_mixture_flux == pytest.approx(physical_boundary_flux, abs=1e-12)


def test_picard_rejects_nonfinite_solution() -> None:
    model = _injection_test_model()
    model.IC = ngs.GridFunction(model.fes)
    model.IC.components[3].Set(0.8)
    model.nonlinear_max_iters = 1
    model.relax_factors = [1.0] * 4
    model.slope_limiter = False
    model.apply_dirichlet_bcs_to = lambda *_args: None

    def nonfinite_solve(_a, _L, _preconditioner, gfu):
        gfu.vec.FV().NumPy()[:] = float('nan')

    model.linear_solve = nonfinite_solve

    class Form:
        def Assemble(self):
            pass

    gfu = ngs.GridFunction(model.fes)
    gfu.vec.data = model.IC.vec
    assert not model.solve_single_step([Form()], [Form()], [None], gfu)
