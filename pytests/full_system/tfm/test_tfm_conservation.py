import math
from pathlib import Path

import ngsolve as ngs
import pytest

from opencmp.helpers.math import Max, Min, tanh
from opencmp.config_functions import ConfigParser
from opencmp.models import get_model_class
from opencmp.solvers import get_solver_class
from import_functions import (_div_tensor, _div_vector, _grad_scalar,
                              exact_solution, ime_config, set_active_ime)


def _integrate_vector(field, mesh):
    return tuple(ngs.Integrate(field[i], mesh) for i in range(mesh.dim))


def _physical_ime_forces(mechanism):
    uc, ud, _, ac = exact_solution()
    ad = 1 - ac
    rho_c = rho_d = dp = 1.0
    nu_c = 0.01
    c_vm = 0.2
    cdis = 0.1
    relative = ud - uc
    speed = ngs.Norm(relative)
    re = speed * dp / nu_c
    cd = ngs.IfPos(
        re,
        Max(Min(24 / re * (1 + 0.15 * re**0.687), 72 / re), 0.0),
        ngs.CoefficientFunction(0.0),
    )

    per_mass_d = ngs.CoefficientFunction((0.0, 0.0))
    per_mass_c = ngs.CoefficientFunction((0.0, 0.0))

    if mechanism == 'drag':
        per_mass_d = 0.75 * cd * rho_c / (rho_d * dp) * speed * relative
        per_mass_c = -0.75 * cd * ad / (ac * dp) * speed * relative
    elif mechanism == 'laminar_dispersion':
        hindered = 1 - 1.166 * ad + 0.5 * ad**2
        per_mass_d = (0.75 * cd * cdis * rho_c / rho_d
                      * hindered * speed**2 * _grad_scalar(ac))
        per_mass_c = (-0.75 * cd * cdis * ad / ac
                      * hindered * speed**2 * _grad_scalar(ac))
    elif mechanism == 'virtual_mass':
        acceleration_difference = (
            _div_tensor(ngs.OuterProduct(ud, ud)) - ud * _div_vector(ud)
            - _div_tensor(ngs.OuterProduct(uc, uc)) + uc * _div_vector(uc)
        )
        per_mass_d = rho_c * c_vm / rho_d * acceleration_difference
        per_mass_c = -ad * c_vm / ac * acceleration_difference
    elif mechanism == 'lift':
        cl = Min(0.288 * tanh(0.121 * re), 0.474)
        grad_uc = ngs.CoefficientFunction((
            (uc[0].Diff(ngs.x), uc[1].Diff(ngs.x)),
            (uc[0].Diff(ngs.y), uc[1].Diff(ngs.y)),
        ), dims=(2, 2))
        curl_uc = grad_uc[1] - grad_uc[2]
        lift_direction = ngs.CoefficientFunction((relative[1], -relative[0]))
        per_mass_d = cl * rho_c / rho_d * curl_uc * lift_direction
        per_mass_c = -cl * ad / ac * curl_uc * lift_direction
    else:
        raise ValueError(mechanism)

    return ad * rho_d * per_mass_d, ac * rho_c * per_mass_c


def test_global_mass_conservation(capsys):
    mesh = ngs.Mesh('pytests/mesh_files/unit_square_coarse.vol')
    for _ in range(3):
        mesh.Refine()

    uc, ud, _, ac = exact_solution()
    ad = 1 - ac
    normal = ngs.specialcf.normal(mesh.dim)

    dispersed_source = _div_vector(ad * ud)
    mixture_source = _div_vector(ac * uc) + _div_vector(ad * ud)

    dispersed_volume = ngs.Integrate(dispersed_source, mesh)
    dispersed_boundary = ngs.Integrate(ad * ud * normal, mesh, ngs.BND)
    mixture_volume = ngs.Integrate(mixture_source, mesh)
    mixture_boundary = ngs.Integrate((ac * uc + ad * ud) * normal,
                                     mesh, ngs.BND)

    dispersed_residual = dispersed_volume - dispersed_boundary
    mixture_residual = mixture_volume - mixture_boundary
    print('TFM global dispersed mass residual: {:.6e}'.format(dispersed_residual))
    print('TFM global mixture mass residual: {:.6e}'.format(mixture_residual))

    assert dispersed_residual == pytest.approx(0.0, abs=1e-11)
    assert mixture_residual == pytest.approx(0.0, abs=1e-11)


def test_discrete_all_ime_global_mass_conservation(tmp_path: Path):
    ime = ('drag', 'laminar_dispersion', 'virtual_mass', 'lift')
    set_active_ime(ime)
    mesh = ngs.Mesh('pytests/mesh_files/unit_square_coarse.vol')
    for _ in range(2):
        mesh.Refine()
    mesh_file = tmp_path / 'unit_square_refined.vol'
    mesh.ngmesh.Save(str(mesh_file))

    config = ConfigParser('pytests/full_system/tfm/config')
    config.set('MESH', 'filename', str(mesh_file))
    config.set('TFM', 'IME', ime_config(ime))
    config.set('TFM', 'lift_wall_deactivation', 'False')
    solver = get_solver_class(config)(get_model_class('TwoFluidModel', False),
                                      config)

    exact = exact_solution(0.0)
    for name, component in solver.model.model_components.items():
        solver.model.UIter.components[component].Set(exact[component])
    solver._create_linear_and_bilinear_forms()
    solver._create_preconditioners()
    solver._assemble()
    solver.gfu.vec.data = solver.model.IC.vec
    solver._apply_boundary_conditions()
    solver.model.linear_solve(solver.a[0], solver.L[0],
                              solver.preconditioners[0], solver.gfu)

    algebraic_residual = solver.L[0].vec.CreateVector()
    algebraic_residual.data = (solver.L[0].vec
                               - solver.a[0].mat * solver.gfu.vec)
    pressure_component = solver.model.model_components['p']
    pressure_dofs = solver.model.fes.Range(pressure_component)
    mixture_weak_residual = algebraic_residual[pressure_dofs].Norm()

    uc = solver.gfu.components[solver.model.model_components['u_c']]
    ud = solver.gfu.components[solver.model.model_components['u_d']]
    ac = solver.gfu.components[solver.model.model_components['alpha_c']]
    normal = ngs.specialcf.normal(solver.model.mesh.dim)

    mixture_source = solver.model.f['p'][0]
    alpha_source = solver.model.f['alpha_c'][0]
    mixture_strong_defect = (
        exact[3] * ngs.div(uc) + _grad_scalar(exact[3]) * uc
        + ngs.div(ud) - exact[3] * ngs.div(ud)
        - _grad_scalar(exact[3]) * ud - mixture_source
    )
    mixture_strong_l2 = math.sqrt(ngs.Integrate(
        mixture_strong_defect**2, solver.model.mesh))
    mixture_residual = (
        ngs.Integrate(mixture_source, solver.model.mesh)
        - ngs.Integrate((exact[3] * uc + (1 - exact[3]) * ud) * normal,
                        solver.model.mesh, ngs.BND)
    )
    dispersed_residual = (
        ngs.Integrate(-alpha_source, solver.model.mesh)
        - ngs.Integrate((1 - ac) * exact[1] * normal,
                        solver.model.mesh, ngs.BND)
    )
    print('TFM discrete all-IME mixture mass residual: {:.6e}'.format(
        mixture_residual))
    print('TFM discrete all-IME dispersed mass residual: {:.6e}'.format(
        dispersed_residual))
    print('TFM discrete all-IME mixture weak residual: {:.6e}'.format(
        mixture_weak_residual))
    print('TFM discrete all-IME mixture strong L2 defect: {:.6e}'.format(
        mixture_strong_l2))
    assert mixture_residual == pytest.approx(0.0, abs=1e-10)
    assert dispersed_residual == pytest.approx(0.0, abs=1e-10)
    assert mixture_weak_residual < 1e-10


@pytest.mark.parametrize('mechanism',
                         ['drag', 'laminar_dispersion', 'virtual_mass', 'lift'])
def test_ime_action_reaction_conservation(mechanism):
    mesh = ngs.Mesh('pytests/mesh_files/unit_square_coarse.vol')
    force_d, force_c = _physical_ime_forces(mechanism)
    residual = _integrate_vector(force_d + force_c, mesh)
    print('TFM {} action-reaction residual: ({:.6e}, {:.6e})'.format(
        mechanism, residual[0], residual[1]))
    assert math.hypot(*residual) < 1e-12


def test_combined_ime_action_reaction_conservation():
    mesh = ngs.Mesh('pytests/mesh_files/unit_square_coarse.vol')
    total = ngs.CoefficientFunction((0.0, 0.0))
    for mechanism in ('drag', 'laminar_dispersion', 'virtual_mass', 'lift'):
        force_d, force_c = _physical_ime_forces(mechanism)
        total += force_d + force_c
    residual = _integrate_vector(total, mesh)
    print('TFM combined IME action-reaction residual: ({:.6e}, {:.6e})'.format(
        residual[0], residual[1]))
    assert math.hypot(*residual) < 1e-12
