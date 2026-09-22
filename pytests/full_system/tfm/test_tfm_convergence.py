import math
from pathlib import Path

import ngsolve as ngs
import pytest

from opencmp.config_functions import ConfigParser
from opencmp.models import get_model_class
from opencmp.solvers import get_solver_class
from import_functions import exact_solution, ime_config, set_active_ime


def _l2_errors(solution, model, time):
    exact = exact_solution(time)
    errors = {}
    for name, component in model.model_components.items():
        difference = solution.components[component] - exact[component]
        if name == 'p':
            difference -= ngs.Integrate(difference, model.mesh) / ngs.Integrate(1, model.mesh)
        errors[name] = math.sqrt(ngs.Integrate(difference * difference, model.mesh))
    uc = solution.components[model.model_components['u_c']]
    ud = solution.components[model.model_components['u_d']]
    ac = exact[3]
    grad_ac = ngs.CoefficientFunction((ac.Diff(ngs.x), ac.Diff(ngs.y)))
    mixture_defect = (
        ac * ngs.div(uc) + grad_ac * uc
        + ngs.div(ud) - ac * ngs.div(ud) - grad_ac * ud
        - model.f['p'][0]
    )
    errors['mixture_mass'] = math.sqrt(ngs.Integrate(
        mixture_defect**2, model.mesh))
    return errors


def _solve_about_exact_picard_state(solver):
    exact = exact_solution(0.0)
    for name, component in solver.model.model_components.items():
        solver.model.UIter.components[component].Set(exact[component])

    solver._create_linear_and_bilinear_forms()
    solver._create_preconditioners()
    solver._assemble()
    solver.gfu.vec.data = solver.gfu_0_list[0].vec
    solver._apply_boundary_conditions()
    solver.model.linear_solve(solver.a[0], solver.L[0],
                              solver.preconditioners[0], solver.gfu)
    return solver.gfu


@pytest.mark.parametrize('ime', [
    (),
    ('drag',),
    ('drag', 'laminar_dispersion'),
    ('lift',),
    ('virtual_mass',),
    ('drag', 'laminar_dispersion', 'virtual_mass', 'lift'),
], ids=['no-ime', 'drag', 'drag-dispersion', 'lift', 'virtual-mass', 'all-ime'])
def test_tfm_manufactured_solution_h_convergence(tmp_path: Path, ime) -> None:
    set_active_ime(ime)
    case_name = '+'.join(ime) if ime else 'no-ime'
    errors = []
    element_counts = []
    base_mesh = 'pytests/mesh_files/unit_square_coarse.vol'

    for level in range(1, 5):
        mesh = ngs.Mesh(base_mesh)
        for _ in range(level):
            mesh.Refine()
        mesh_file = tmp_path / ('unit_square_refined_{}.vol'.format(level))
        mesh.ngmesh.Save(str(mesh_file))

        config = ConfigParser('pytests/full_system/tfm/config')
        config.set('MESH', 'filename', str(mesh_file))
        config.set('TFM', 'IME', ime_config(ime))
        config.set('TFM', 'lift_wall_deactivation', 'False')
        solver = get_solver_class(config)(get_model_class('TwoFluidModel', False), config)
        solver.gfu_0_list[0].vec.data = solver.model.IC.vec
        solution = _solve_about_exact_picard_state(solver)
        errors.append(_l2_errors(solution, solver.model, 0.0))
        element_counts.append(solver.model.mesh.ne)

    for variable in ('u_c', 'u_d', 'p', 'alpha_c', 'mixture_mass'):
        assert errors[-1][variable] < errors[0][variable]
        rates = [
            math.log(errors[level][variable] / errors[level + 1][variable])
            / math.log(math.sqrt(element_counts[level + 1] / element_counts[level]))
            for level in range(len(errors) - 1)
        ]
        print('TFM MMS [{}] {}: errors={}, orders={}'.format(
            case_name, variable,
            ['{:.6e}'.format(level[variable]) for level in errors],
            ['{:.3f}'.format(rate) for rate in rates]))
        # The strong mixture residual contains derivatives of the HDiv
        # velocity error and therefore converges one order below the velocity
        # L2 error. All solution variables should exceed first order; the
        # strong mixture defect should approach its expected first order.
        minimum_order = 0.9 if variable == 'mixture_mass' else 1.0
        assert min(rates[-2:]) > minimum_order, (variable, errors, rates)
