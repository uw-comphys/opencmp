import ngsolve as ngs
from opencmp.helpers.math import Max, Min, tanh
from opencmp.models.tfm import TwoFluidModel


ACTIVE_IME = frozenset()


def set_active_ime(ime):
    global ACTIVE_IME
    ACTIVE_IME = frozenset(ime)


def ime_config(ime):
    """Build a [TFM] IME value from mechanism names, using each one's first closure model."""
    return '\n'.join('{} -> {}'.format(mechanism, TwoFluidModel.IME_MODELS[mechanism][0])
                     for mechanism in ime)


def exact_solution(t=None):
    k = ngs.pi
    uc = ngs.CoefficientFunction((
        1.0 - 0.01 * ngs.cos(k * ngs.x) * ngs.sin(k * ngs.y),
        0.01 * ngs.sin(k * ngs.x) * ngs.cos(k * ngs.y),
    ))
    ud = ngs.CoefficientFunction((
        1.0 + 0.01 * ngs.sin(k * ngs.x) * ngs.sin(k * ngs.y),
        0.01 * ngs.cos(k * ngs.x) * ngs.cos(k * ngs.y),
    ))
    p = -0.0025 * (ngs.cos(2 * k * ngs.x) + ngs.cos(2 * k * ngs.y))
    alpha_c = 0.9 + 0.002 * ngs.sin(k * ngs.x + ngs.pi / 4) * ngs.cos(k * ngs.y)
    return uc, ud, p, alpha_c


def _value(t_param, time_step, component):
    return exact_solution(t_param[time_step])[component]


def exact_uc(t_param, model_variables, mesh, time_step):
    return _value(t_param, time_step, 0)


def exact_ud(t_param, model_variables, mesh, time_step):
    return _value(t_param, time_step, 1)


def exact_p(t_param, model_variables, mesh, time_step):
    return _value(t_param, time_step, 2)


def exact_alpha_c(t_param, model_variables, mesh, time_step):
    return _value(t_param, time_step, 3)


def _div_vector(v):
    return v[0].Diff(ngs.x) + v[1].Diff(ngs.y)


def _grad_scalar(s):
    return ngs.CoefficientFunction((s.Diff(ngs.x), s.Diff(ngs.y)))


def _grad_vector(v):
    return ngs.CoefficientFunction((
        (v[0].Diff(ngs.x), v[1].Diff(ngs.x)),
        (v[0].Diff(ngs.y), v[1].Diff(ngs.y)),
    ), dims=(2, 2))


def _div_tensor(T):
    return ngs.CoefficientFunction((
        T[0].Diff(ngs.x) + T[2].Diff(ngs.y),
        T[1].Diff(ngs.x) + T[3].Diff(ngs.y),
    ))


def source_alpha_c(t_param, model_variables, mesh, time_step):
    _, ud, _, alpha_c = exact_solution()
    return _div_vector(alpha_c * ud) - _div_vector(ud)


def source_mixture_mass(t_param, model_variables, mesh, time_step):
    uc, ud, _, alpha_c = exact_solution()
    return _div_vector(alpha_c * uc) + _div_vector((1 - alpha_c) * ud)


def _momentum_source(phase):
    uc, ud, p, alpha_c = exact_solution()
    if phase == 'c':
        u, rho, nu, alpha = uc, 1.0, 0.01, alpha_c
    else:
        u, rho, nu, alpha = ud, 1.0, 0.01, 1 - alpha_c
    convection = _div_tensor(ngs.OuterProduct(u, u)) - u * _div_vector(u)
    source = convection + _grad_scalar(p) / rho
    grad_u = _grad_vector(u)
    identity = ngs.CoefficientFunction(((1, 0), (0, 1)), dims=(2, 2))
    stress = grad_u + grad_u.trans - (2.0 / 3.0) * _div_vector(u) * identity
    alpha_denominator = alpha if phase == 'c' else alpha + 1e-5
    source -= nu * (_div_tensor(stress)
                    + stress.trans * _grad_scalar(alpha) / alpha_denominator)
    gravity = ngs.CoefficientFunction((0.0, -9.81))
    source -= gravity

    relative_velocity = ud - uc
    relative_speed = ngs.Norm(relative_velocity)
    rho_c = rho_d = 1.0
    dp = 1.0
    c_vm = 0.2
    cdis = 0.1
    ad = 1 - alpha_c

    re = relative_speed * dp / 0.01
    cd = ngs.IfPos(
        re,
        Max(Min(24 / re * (1 + 0.15 * re**0.687), 72 / re), 0.0),
        ngs.CoefficientFunction(0.0),
    )

    if 'drag' in ACTIVE_IME:
        if phase == 'd':
            source += 0.75 * cd * rho_c / (rho_d * dp) * relative_speed * relative_velocity
        else:
            source += -0.75 * cd * ad / (alpha_c * dp) * relative_speed * relative_velocity

    acceleration_difference = (
        _div_tensor(ngs.OuterProduct(ud, ud)) - ud * _div_vector(ud)
        - _div_tensor(ngs.OuterProduct(uc, uc)) + uc * _div_vector(uc)
    )
    if 'virtual_mass' in ACTIVE_IME:
        coefficient = (rho_c * c_vm / rho_d if phase == 'd'
                       else -ad * c_vm / alpha_c)
        source += coefficient * acceleration_difference

    if 'lift' in ACTIVE_IME:
        # Tomiyama coefficient with Eo=0 for this equal-density MMS case.
        cl = Min(0.288 * tanh(0.121 * re), 0.474)
        curl_uc = _grad_vector(uc)[1] - _grad_vector(uc)[2]
        lift_vector = ngs.CoefficientFunction((relative_velocity[1],
                                               -relative_velocity[0]))
        coefficient = (rho_c / rho_d if phase == 'd' else -ad / alpha_c)
        source += cl * coefficient * curl_uc * lift_vector

    if 'laminar_dispersion' in ACTIVE_IME:
        hindered = 1 - 1.166 * ad + 0.5 * ad**2
        coefficient = (0.75 * cd * cdis * rho_c / rho_d * hindered * relative_speed**2
                       if phase == 'd'
                       else -0.75 * cd * cdis * ad / alpha_c * hindered * relative_speed**2)
        # Dispersion is assembled on the linear-form side in TwoFluidModel.
        source -= coefficient * _grad_scalar(alpha_c)

    return source


def source_uc(t_param, model_variables, mesh, time_step):
    return _momentum_source('c')


def source_ud(t_param, model_variables, mesh, time_step):
    return _momentum_source('d')
