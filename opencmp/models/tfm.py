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

import logging
from typing import Dict, List, Optional, Union

import ngsolve as ngs
from ngsolve import GridFunction, FESpace, BilinearForm, LinearForm, Preconditioner, Parameter
from ngsolve.comp import ProxyFunction

from . import Model
from ..helpers.dg import avg, jump, grad_avg, weighted_grad_avg, weighted_div_avg
from ..helpers.math import tanh, Max, Min
from ..helpers.ngsolve_ import get_special_functions, curl_3d
from ..helpers.limiter import Limiter
from ..helpers.error import norm, mean
from ..helpers.wall_func import wall_distance


class TwoFluidModel(Model):
    """
    Laminar Euler-Euler two-fluid model for two-phase pipe flows.

    Implements four coupled equations: u_c (HDiv/BDM), u_d (HDiv/BDM),
    p (L2), alpha_c (L2).  The nonlinear system is resolved by Picard
    (fixed-point) iteration each time step.  Turbulence closure is left to a
    subclass (cf. INS / KEpsilonINS).
    """

    # Interphase momentum exchange mechanisms and the closure models available to each.
    IME_MODELS = {'drag': ('Tomiyama', 'SchillerNaumann'),
                  'lift': ('Tomiyama', 'LegendreMagnaudet'),
                  'virtual_mass': ('ConstantCoefficient',),
                  'laminar_dispersion': ('ConstantCoefficient',)}

    # ------------------------------------------------------------------
    # Abstract-method overrides — bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def allows_explicit_schemes() -> bool:
        return False

    def _define_model_components(self) -> Dict[str, Optional[int]]:
        return {'u_c': 0, 'u_d': 1, 'p': 2, 'alpha_c': 3}

    def _define_model_local_error_components(self) -> Dict[str, bool]:
        return {'u_c': True, 'u_d': True, 'p': False, 'alpha_c': True}

    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        return [{'u_c': True, 'u_d': True, 'p': False, 'alpha_c': True}]

    def _define_num_weak_forms(self) -> int:
        return 1

    def _define_bc_types(self) -> List[str]:
        return ['dirichlet', 'zero_stress', 'zero_gradient', 'zero_backflow', 'slip']

    # ------------------------------------------------------------------
    # Phase 1 — lifecycle hooks
    # ------------------------------------------------------------------

    def _tfm_option(self, key: str, val_type, default):
        """Read an optional model-specific setting from [TFM]."""
        if not self.config.has_option('TFM', key):
            return default
        try:
            return self.config.get_item(['TFM', key], val_type, quiet=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid [TFM] value for '{}'.".format(key)) from exc

    def _other_option(self, key: str, val_type, default):
        """Read an optional generic numerical setting from [OTHER]."""
        if not self.config.has_option('OTHER', key):
            return default
        return self.config.get_item(['OTHER', key], val_type, quiet=True)

    def _parse_ime(self) -> Dict[str, str]:
        """Read [TFM] IME, which maps each active mechanism to its closure model."""
        if not self.config.has_option('TFM', 'ime'):
            return {'drag': 'Tomiyama'}
        if not self.config['TFM']['ime'].strip():
            return {}

        ime = {key.lower(): value for key, value in
               self.config.get_dict(['TFM', 'ime'], '', None, all_str=True).items()}

        unknown_ime = set(ime) - set(self.IME_MODELS)
        if unknown_ime:
            raise ValueError('Unknown [TFM] IME mechanism(s): {}.'.format(', '.join(sorted(unknown_ime))))
        for mechanism, model in ime.items():
            if model not in self.IME_MODELS[mechanism]:
                raise ValueError("[TFM] IME '{}' must use one of: {}.".format(
                    mechanism, ', '.join(self.IME_MODELS[mechanism])))
        if 'laminar_dispersion' in ime and 'drag' not in ime:
            raise ValueError("[TFM] IME 'laminar_dispersion' requires 'drag'.")

        return ime

    def _pre_init(self) -> None:
        allowed_keys = {'canonical_form', 'ime', 'lift_wall_deactivation', 'lift_wall_boundaries'}
        unknown_keys = set(self.config['TFM']) - allowed_keys if self.config.has_section('TFM') else set()
        if unknown_keys:
            raise ValueError('Unknown [TFM] option(s): {}.'.format(', '.join(sorted(unknown_keys))))

        self.canonical_form = self._tfm_option('canonical_form', str, 'B-TFM')
        if self.canonical_form not in ('B-TFM', 'C-TFM'):
            raise ValueError("[TFM] canonical_form must be 'B-TFM' or 'C-TFM'.")

        self.slope_limiter = self._other_option('slope_limiter', bool, True)

        ime = self._parse_ime()
        self.drag_switch = 'drag' in ime
        self.VM_switch = 'virtual_mass' in ime
        self.Disp_switch = 'laminar_dispersion' in ime
        self.Lift_switch = 'lift' in ime

        self.drag_model = ime.get('drag', 'Tomiyama')
        self.lift_model = ime.get('lift', 'Tomiyama')

        # Artificial diffusion on the alpha_c transport.
        self.diffusion_switch = self._other_option('diffusion_switch', bool, False)
        self.mean_zero_pressure = self._other_option('mean_zero_pressure', bool, False)

        # Lift wall-deactivation taper (off by default; costs one wall-distance solve).
        self.lift_wall_deactivation = self._tfm_option('lift_wall_deactivation', bool, False)
        self.lift_wall_boundaries = self._tfm_option('lift_wall_boundaries', str, '')
        if self.lift_wall_deactivation and not self.Lift_switch:
            raise ValueError('lift_wall_deactivation requires lift in [TFM] IME.')

        # Loaded from model_dir/model_config after ModelFunctions is initialized.
        self.injection_switch = False
        self.injection_region = ''
        self.inj_mass_flowrate = 0.0
        self.inj_velocity = 0.0

    def _velocity_space_boundaries(self, var: str) -> str:
        """Combine full Dirichlet and normal-only slip constraints for H(div)."""
        dirichlet = list(self.BC.get('dirichlet', {}).get(var, {}))
        slip = list(self.BC.get('slip', {}).get(var, {}))
        overlap = set(dirichlet) & set(slip)
        if overlap:
            raise ValueError(
                "Boundary marker(s) {} cannot be both DIRICHLET and SLIP for '{}'."
                .format(', '.join(sorted(overlap)), var)
            )
        return '|'.join(dirichlet + slip)

    def _construct_fes(self) -> FESpace:
        # The weak form assumes H(div)-conforming velocities (the mixture mass
        # conservation and the UDS momentum fluxes are written on u.n) and a
        # discontinuous pressure, so those two are constrained rather than free.
        for name in ('u_c', 'u_d'):
            if self.element[name] != 'HDiv':
                raise ValueError("TwoFluidModel requires an HDiv element for '{}', got '{}'."
                                 .format(name, self.element[name]))
        if self.element['p'] != 'L2':
            raise ValueError("TwoFluidModel requires an L2 element for 'p', got '{}'."
                             .format(self.element['p']))

        scalar_ord = max(self.interp_ord - 1, 0)
        # HDiv strongly constrains only the normal trace. DIRICHLET markers also
        # receive the weak full-vector terms below; SLIP markers do not, leaving
        # zero tangential traction as their natural boundary condition.
        uc_dirichlet = self._velocity_space_boundaries('u_c')
        ud_dirichlet = self._velocity_space_boundaries('u_d')
        fes_uc = ngs.HDiv(self.mesh, order=self.interp_ord,
                          dirichlet=uc_dirichlet, dgjumps=self.DG)
        fes_ud = ngs.HDiv(self.mesh, order=self.interp_ord,
                          dirichlet=ud_dirichlet, dgjumps=self.DG)
        fes_p  = ngs.L2(self.mesh, order=scalar_ord, dgjumps=self.DG)
        # alpha_c is an ordinary scalar transport variable -- any element the user asks for.
        # No `dirichlet` here: its BCs are imposed weakly through the UDS flux terms.
        fes_ac = getattr(ngs, self.element['alpha_c'])(self.mesh, order=scalar_ord, dgjumps=self.DG)
        spaces = [fes_uc, fes_ud, fes_p, fes_ac]
        if getattr(self, 'mean_zero_pressure', False):
            spaces.append(ngs.NumberSpace(self.mesh))
        return FESpace(spaces, dgjumps=self.DG)

    def _set_model_parameters(self) -> None:
        p = self.model_functions.model_parameters_dict
        self.rho_c   = p['rho_c']['all']
        self.rho_d   = p['rho_d']['all']
        self.nu_c    = p['nu_c']['all']
        self.nu_d    = p['nu_d']['all']
        self.sigma_c = p['sigma_c']['all']
        self.dp      = p['dp']['all']
        self.C_VM    = p['c_vm']['all']
        self.Cdis    = p['cdis']['all']
        self.D_art   = p['d_artificial']['all']
        self.f = self.model_functions.model_functions_dict.get('source', {})

        model_config = self.model_functions.config
        self.injection_switch = model_config.has_section('INJECTION')
        if self.injection_switch:
            allowed = {'region', 'mass_flow_rate', 'velocity'}
            unknown = set(model_config['INJECTION']) - allowed
            if unknown:
                raise ValueError('Unknown model [INJECTION] option(s): {}.'
                                 .format(', '.join(sorted(unknown))))
            for required in allowed:
                if not model_config.has_option('INJECTION', required):
                    raise ValueError("model_dir/model_config [INJECTION] requires '{}'."
                                     .format(required))
            self.injection_region = model_config.get_item(['INJECTION', 'region'], str)
            self.inj_mass_flowrate = model_config.get_item(['INJECTION', 'mass_flow_rate'], float)
            self.inj_velocity = model_config.get_item(['INJECTION', 'velocity'], float)

        if self.mesh.dim == 2:
            self.gravity = ngs.CoefficientFunction((0.0, -9.81))
        else:
            self.gravity = ngs.CoefficientFunction((0.0, -9.81, 0.0))

    def _post_init(self) -> None:
        self._validate_bc_variables()
        self.nonlinear = True
        self.linearize = 'Picard'

        nonlinear_tolerance = self.config.get_dict(
            ['SOLVER', 'nonlinear_tolerance'], self.run_dir, None)
        self.abs_nonlinear_tolerance = nonlinear_tolerance['absolute']
        self.rel_nonlinear_tolerance = nonlinear_tolerance['relative']
        self.nonlinear_max_iters = self.config.get_item(
            ['SOLVER', 'nonlinear_max_iterations'], int)
        if self.nonlinear_max_iters < 1:
            raise ValueError('nonlinear_max_iterations must be >= 1.')

        try:
            relax = self.config.get_list(['SOLVER', 'relaxation_factors'], float)
        except Exception:
            relax = []
        n_comp = len(self.model_components)
        self.relax_factors = relax if len(relax) == n_comp else [1.0] * n_comp

        # Picard iterate (updated at the start of every Picard sub-iteration)
        self.UIter = ngs.GridFunction(self.fes)
        self.UIter.vec.data = self.IC.vec

        # Piecewise-constant closure GFs (live outside self.fes)
        _cl_fes = ngs.L2(self.mesh, order=0)
        self.Cd_gfu = ngs.GridFunction(_cl_fes)
        self.Cl_gfu = ngs.GridFunction(_cl_fes)

        # Special DG functions (self.nu = ipc * interp_ord^2 is the penalty parameter).
        self.n, self.h, self.penalty_interior, self.IM = get_special_functions(self.mesh, self.nu)

        # Slope limiter instance
        self._lim = Limiter(self.mesh)

        # Wall distance for the lift taper. Solved once (Eikonal Newton solve), and
        # only when the taper is actually on.
        if self.lift_wall_deactivation:
            if not self.lift_wall_boundaries:
                raise ValueError(
                    'lift_wall_deactivation requires lift_wall_boundaries to be specified.'
                )
            self._wall_dist = wall_distance(self.mesh, self.lift_wall_boundaries)
        else:
            self._wall_dist = None

    # ------------------------------------------------------------------
    # Internal helpers — closure coefficients
    # ------------------------------------------------------------------

    def _bc_regex(self, bc_type: str, varname: str) -> str:
        return '|'.join(self.BC.get(bc_type, {}).get(varname, {}).keys())

    # Each outflow condition applies to exactly one kind of variable.
    BC_VARIABLES = {'zero_stress': ('u_c', 'u_d'),
                    'zero_gradient': ('alpha_c',),
                    'zero_backflow': ('alpha_c',)}

    def _validate_bc_variables(self) -> None:
        for bc_type, allowed in self.BC_VARIABLES.items():
            for var in self.BC.get(bc_type, {}):
                if var not in allowed:
                    raise ValueError(
                        "[{}] is only meaningful for {}, not '{}'."
                        .format(bc_type.upper(), ' and '.join(allowed), var))

        overlap = set(self.BC.get('zero_gradient', {}).get('alpha_c', {})) \
            & set(self.BC.get('zero_backflow', {}).get('alpha_c', {}))
        if overlap:
            raise ValueError(
                "Boundary marker(s) {} cannot be both ZERO_GRADIENT and ZERO_BACKFLOW "
                "for 'alpha_c'.".format(', '.join(sorted(overlap))))

    def time_derivative_terms(self, gfu_lst: List[List[GridFunction]], scheme: str,
                              step: int = 1):
        """Add the virtual-mass cross derivatives to the standard time terms."""
        a, L = super().time_derivative_terms(gfu_lst, scheme, step)
        if not self.VM_switch:
            return a, L

        U, V = self.get_trial_and_test_functions()
        comp = self.model_components
        uc, ud = U[comp['u_c']], U[comp['u_d']]
        vc, vd = V[comp['u_c']], V[comp['u_d']]

        Ac = self.UIter.components[comp['alpha_c']]
        Ad = 1 - Ac
        VM_d = self.rho_c[0] * self.C_VM[0] / self.rho_d[0]
        VM_c = -Ad / Ac * self.C_VM[0]

        if scheme in ('explicit euler', 'implicit euler', 'crank nicolson', 'adaptive imex pred'):
            current_coefficient = 1.0
            old = gfu_lst[1]
            old_difference = old[comp['u_d']] - old[comp['u_c']]
        elif scheme == 'CNLF':
            current_coefficient = 1.0
            old = gfu_lst[2]
            old_difference = old[comp['u_d']] - old[comp['u_c']]
        elif scheme == 'SBDF':
            current_coefficient = 11.0 / 6.0
            old_difference = (
                3.0 * (gfu_lst[1][comp['u_d']] - gfu_lst[1][comp['u_c']])
                - 1.5 * (gfu_lst[2][comp['u_d']] - gfu_lst[2][comp['u_c']])
                + (1.0 / 3.0) * (gfu_lst[3][comp['u_d']] - gfu_lst[3][comp['u_c']])
            )
        elif scheme in ('RK 222', 'RK 232'):
            current_coefficient = 1.0
            old = gfu_lst[step]
            old_difference = old[comp['u_d']] - old[comp['u_c']]
        else:
            raise ValueError('Scheme "{}" is not implemented'.format(scheme))

        current_difference = ud - uc
        a[0] += current_coefficient * (
            VM_d * current_difference * vd + VM_c * current_difference * vc)
        L[0] += VM_d * old_difference * vd + VM_c * old_difference * vc
        return a, L

    def _get_drag_coeff(self, wd, wc, Ad, ts: int) -> ngs.CoefficientFunction:
        rho_c = self.rho_c[ts]; rho_d = self.rho_d[ts]
        nu_c  = self.nu_c[ts];  dp    = self.dp[ts]
        sigma_c = self.sigma_c[ts]
        g_mag = ngs.Norm(self.gravity)
        Eo = g_mag * (rho_c - rho_d) * dp**2 / sigma_c
        Re = ngs.Norm(wd - wc) * dp / nu_c
        if self.drag_model == 'Tomiyama':
            Cd = ngs.IfPos(Re,
                           Max(Min(24/Re*(1 + 0.15*Re**0.687), 72/Re),
                                8/3*Eo/(Eo + 4)),
                           ngs.CoefficientFunction(0.0))
        else:
            Cd = ngs.IfPos(Re - 1000,
                           ngs.CoefficientFunction(0.44),
                           24/Re*(1 + 0.15*Re**0.687))
        return Cd

    def _get_lift_coeff(self, wd, wc, ts: int) -> ngs.CoefficientFunction:
        rho_c = self.rho_c[ts]; rho_d = self.rho_d[ts]
        nu_c  = self.nu_c[ts];  dp    = self.dp[ts]
        sigma_c = self.sigma_c[ts]
        g_mag = ngs.Norm(self.gravity)
        Eo = g_mag * (rho_c - rho_d) * dp**2 / sigma_c
        Re = ngs.Norm(wd - wc) * dp / nu_c
        Sr = dp**2 / (Re * nu_c + 1e-30) * ngs.Norm(ngs.grad(wc))
        if self.lift_model == 'LegendreMagnaudet':
            # Guarded denominator: at zero slip Re and Sr both vanish, giving 0/0.
            ClLow = (6*2.255)**2 * Sr**2 / (ngs.pi**4 * Re * (Sr + 0.2*Re)**3 + 1e-10)
            ClHigh = (0.5 * (Re + 16) / (Re + 29))**2
            Cl = (ClLow + ClHigh)**0.5
        else:  # Tomiyama
            fEo = 0.00105*Eo**3 - 0.0159*Eo**2 - 0.0204*Eo + 0.474
            Cl = (ngs.IfPos(Eo - 4, 0, 1) * Min(0.288*tanh(0.121*Re), fEo)
                  + ngs.IfPos(Eo - 4, 1, 0) * ngs.IfPos(10.7 - Eo, 1, 0) * fEo
                  + ngs.IfPos(Eo - 10.7, 1, 0) * (-0.288))

        # Optional near-wall lift taper.
        if self.lift_wall_deactivation:
            dp_factor = 3
            xw = self._wall_dist
            s = 2*xw/(dp_factor*dp) - 1
            activation = (ngs.IfPos(xw - dp_factor*dp, 1, 0)
                          + ngs.IfPos(dp_factor*dp - xw, 1, 0)
                          * ngs.IfPos(xw - 0.5*dp_factor*dp, 1, 0)
                          * (3*s**2 - 2*s**3))
            Cl = Cl * activation
        return Cl

    # ------------------------------------------------------------------
    # Internal helpers — UDS numerical fluxes
    # ------------------------------------------------------------------

    def _NF_UDS_mass(self, u, w, facet: str = 'Interior', bl: bool = True):
        """UDS numerical flux for scalar advection (alpha_c)."""
        n = self.n
        if facet == 'Interior':
            return avg(u) * (w * n) + 0.5 * ngs.Norm(w * n) * jump(u)
        if facet == 'Dirichlet':
            sign = 1.0 if bl else -1.0
            return 0.5 * u * (w * n) + sign * 0.5 * u * ngs.Norm(w * n)
        if facet == 'Neumann':
            return u * Max(w * n, ngs.CoefficientFunction(0.0))
            #return u * w * n

    def _NF_UDS_mom(self, u, w, facet: str = 'Interior', bl: bool = True):
        """UDS numerical flux for momentum (vector) advection."""
        n = self.n
        if facet == 'Interior':
            return (ngs.OuterProduct(avg(u), w)
                    + 0.5 * ngs.Norm(w * n) * ngs.OuterProduct(jump(u), n))
        if facet == 'Dirichlet':
            sign = 1.0 if bl else -1.0
            return (0.5 * ngs.OuterProduct(u, w)
                    + sign * 0.5 * ngs.Norm(w * n) * ngs.OuterProduct(u, n))
        if facet == 'Neumann':
            return ngs.OuterProduct(u, n) * Max(w * n, ngs.CoefficientFunction(0.0))

    # ------------------------------------------------------------------
    # Phase 2 — bilinear form (spatial operator)
    # ------------------------------------------------------------------

    def construct_bilinear_time_ODE(self,
                                    U: Union[List[ProxyFunction], List[GridFunction]],
                                    V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0),
                                    time_step: int = 0) -> List:
        ts   = time_step
        n    = self.n
        IM   = self.IM
        pen  = self.penalty_interior
        comp = self.model_components

        # Trial / test functions
        uc, ud  = U[comp['u_c']], U[comp['u_d']]
        p       = U[comp['p']]
        alpha_c = U[comp['alpha_c']]
        vc, vd  = V[comp['u_c']], V[comp['u_d']]
        q, r    = V[comp['p']], V[comp['alpha_c']]

        # Picard iterate fields
        wc = self.UIter.components[comp['u_c']]
        wd = self.UIter.components[comp['u_d']]
        Ac = self.UIter.components[comp['alpha_c']]
        Ad = 1 - Ac

        # Physical parameters
        rho_c = self.rho_c[ts]; rho_d = self.rho_d[ts]
        nu_c  = self.nu_c[ts];  nu_d  = self.nu_d[ts]
        dp    = self.dp[ts]

        # BC regex strings
        ac_d_reg  = self.dirichlet_names.get('alpha_c', '')
        ac_zg_reg = self._bc_regex('zero_gradient', 'alpha_c')
        ac_zb_reg = self._bc_regex('zero_backflow', 'alpha_c')
        ac_n_reg  = '|'.join(reg for reg in (ac_zg_reg, ac_zb_reg) if reg)
        uc_d_reg  = self.dirichlet_names.get('u_c', '')
        uc_n_reg  = self._bc_regex('zero_stress', 'u_c')
        ud_d_reg  = self.dirichlet_names.get('u_d', '')
        ud_n_reg  = self._bc_regex('zero_stress', 'u_d')

        a = ngs.CoefficientFunction(0.0) * ngs.dx

        # ============================================================
        # 1. Alpha_c (dispersed-phase mass conservation) — a terms only
        #    (a_dt = alpha_c*r*dx is handled by time_derivative_terms)
        # ============================================================
        # div(u_d)*r is treated EXPLICITLY (lagged on the Picard wind w_d) and
        # lives in the linear form as +dt*div(w_d)*r -- see construct_linear.
        a += (dt * -alpha_c * (wd * ngs.grad(r))) * ngs.dx
        a += (dt * jump(r) * self._NF_UDS_mass(alpha_c, wd)) * ngs.dx(skeleton=True)
        if ac_d_reg:
            a += (dt * r * self._NF_UDS_mass(alpha_c, wd, 'Dirichlet', True)) * self._ds(ac_d_reg)
        if ac_zg_reg:
            a += (dt * r * alpha_c * (wd * n)) * self._ds(ac_zg_reg)
        if ac_zb_reg:
            # Outflow only; the inflow half is prescribed in construct_linear.
            a += (dt * r * alpha_c * Max(wd * n, ngs.CoefficientFunction(0.0))) * self._ds(ac_zb_reg)

        # Artificial diffusion (SIPG) on alpha_c -- stabilises sharp fronts.
        if self.diffusion_switch:
            art = self.D_art[ts]
            a += (dt * art * ngs.grad(alpha_c) * ngs.grad(r)) * ngs.dx
            a += (-dt * art * (n * grad_avg(r)) * jump(alpha_c)) * ngs.dx(skeleton=True)
            a += (dt * art * (pen * jump(alpha_c)) * jump(r)) * ngs.dx(skeleton=True)
            a += (-dt * art * (n * grad_avg(alpha_c)) * jump(r)) * ngs.dx(skeleton=True)
            if ac_d_reg:
                a += (-dt * art * alpha_c * (ngs.grad(r) * n)) * self._ds(ac_d_reg)
                a += (dt * art * (pen * alpha_c - ngs.grad(alpha_c) * n) * r) * self._ds(ac_d_reg)
            if ac_n_reg:
                a += (-dt * art * (n * ngs.grad(alpha_c)) * r) * self._ds(ac_n_reg)

        # ============================================================
        # 2. Pressure / mixture mass conservation — NOT dt-scaled
        # ============================================================
        a += (Ac * ngs.div(uc) * q) * ngs.dx
        a += (ngs.grad(Ac) * uc * q) * ngs.dx
        a += (ngs.div(ud) * q) * ngs.dx
        a += (-Ac * ngs.div(ud) * q) * ngs.dx
        a += (-ngs.grad(Ac) * ud * q) * ngs.dx
        if self.mean_zero_pressure:
            pressure_mean, pressure_mean_test = U[-1], V[-1]
            a += (pressure_mean * q + p * pressure_mean_test) * ngs.dx

        # ============================================================
        # 3 & 4. Phase momentum conservation (c and d)
        # ============================================================
        for (u_tr, w_pi, v_ts, A_pi, rho, nu_lam, d_reg, n_reg, phase) in [
            (uc, wc, vc, Ac, rho_c, nu_c, uc_d_reg, uc_n_reg, 'c'),
            (ud, wd, vd, Ad, rho_d, nu_d, ud_d_reg, ud_n_reg, 'd'),
        ]:
            tau = ngs.grad(u_tr) + ngs.grad(u_tr).trans - 2.0/3.0 * ngs.div(u_tr) * IM

            # Advection — bulk (Picard-linearised convection)
            a += (dt * -ngs.div(w_pi) * u_tr * v_ts) * ngs.dx
            a += (dt * -ngs.InnerProduct(ngs.grad(v_ts), ngs.OuterProduct(u_tr, w_pi))) * ngs.dx

            # Pressure (shared mixture pressure p)
            a += (dt * -p / rho * ngs.div(v_ts)) * ngs.dx

            # Advection — interior facets
            a += (dt * ngs.InnerProduct(ngs.OuterProduct(jump(v_ts), n),
                                         self._NF_UDS_mom(u_tr, w_pi))) * ngs.dx(skeleton=True)
            # Advection — Dirichlet boundary (bilinear)
            if d_reg:
                a += (dt * ngs.InnerProduct(ngs.OuterProduct(v_ts, n),
                                             self._NF_UDS_mom(u_tr, w_pi, 'Dirichlet', True))) \
                     * self._ds(d_reg)
            # Advection — Neumann boundary (bilinear)
            if n_reg:
                a += (dt * ngs.InnerProduct(ngs.OuterProduct(v_ts, n),
                                             self._NF_UDS_mom(u_tr, w_pi, 'Neumann', True))) \
                     * self._ds(n_reg)

            # Viscous: B-TFM (phase c only) or C-TFM (both phases)
            do_viscous = (self.canonical_form == 'B-TFM' and phase == 'c') or \
                         (self.canonical_form == 'C-TFM')
            if do_viscous:
                nu = ngs.CoefficientFunction(nu_lam)

                # grad of the phase fraction. A_d = 1 - A_c is a CF expression and
                # ngs.grad() only accepts a GridFunction/proxy, so use grad(A_d) =
                # -grad(A_c) with the actual A_c iterate (also tracks Picard updates).
                grad_A = ngs.grad(Ac) if phase == 'c' else -ngs.grad(Ac)

                if self.canonical_form == 'B-TFM':
                    factor     = nu / A_pi
                    factor_avg = avg(nu / A_pi)
                    a += (dt * -nu * ngs.InnerProduct(
                        tau, ngs.OuterProduct(v_ts, grad_A / (A_pi**2)))) * ngs.dx
                else:  # C-TFM
                    factor     = ngs.CoefficientFunction(nu)
                    factor_avg = avg(ngs.CoefficientFunction(nu))
                    if phase == 'c':
                        a += (dt * -nu * ngs.InnerProduct(
                            tau, ngs.OuterProduct(v_ts, grad_A / A_pi))) * ngs.dx
                    else:
                        a += (dt * -nu * ngs.InnerProduct(
                            tau, ngs.OuterProduct(v_ts,
                                                   grad_A / (A_pi + 1e-5)))) * ngs.dx

                # Bulk viscous
                a += (dt * factor * ngs.InnerProduct(ngs.grad(v_ts), tau)) * ngs.dx

                # Interior IP-DG (SIPG for viscous stress)
                stress_avg = (weighted_grad_avg(u_tr, factor) + weighted_grad_avg(u_tr, factor).trans
                              - 2.0/3.0 * weighted_div_avg(u_tr, factor) * IM)
                a += (dt * -ngs.InnerProduct(stress_avg,
                                              ngs.OuterProduct(jump(v_ts), n))) * ngs.dx(skeleton=True)
                a += (dt * factor_avg * pen
                      * ngs.InnerProduct(ngs.OuterProduct(jump(u_tr), n),
                                          ngs.OuterProduct(jump(v_ts), n))) * ngs.dx(skeleton=True)
                # The reference form uses lower-case grad(v), which is Grad(v)^T
                # for a vector. Transpose the outer product to express that term
                # with OpenCMP's Grad-based helper.
                a += (dt * -ngs.InnerProduct(ngs.OuterProduct(n, jump(u_tr)),
                                              weighted_grad_avg(v_ts, factor))) * ngs.dx(skeleton=True)
                a += (dt * -ngs.InnerProduct(ngs.OuterProduct(jump(u_tr), n),
                                              weighted_grad_avg(v_ts, factor))) * ngs.dx(skeleton=True)

                # Dirichlet boundary — viscous bilinear (Nitsche)
                if d_reg:
                    a += (dt * -factor * ngs.InnerProduct(tau, ngs.OuterProduct(v_ts, n))) \
                         * self._ds(d_reg)
                    a += (dt * factor * pen
                          * ngs.InnerProduct(ngs.OuterProduct(u_tr, n),
                                              ngs.OuterProduct(v_ts, n))) * self._ds(d_reg)
                    a += (dt * -factor * ngs.InnerProduct(
                        ngs.OuterProduct(u_tr, n), ngs.grad(v_ts))) * self._ds(d_reg)
                    a += (dt * -factor * ngs.InnerProduct(
                        ngs.OuterProduct(u_tr, n), ngs.grad(v_ts).trans)) * self._ds(d_reg)

        # ============================================================
        # 5. Interphase closures — bilinear contributions
        # ============================================================

        # — Drag (dt-scaled)
        if self.drag_switch:
            Cd = self._get_drag_coeff(wd, wc, Ad, ts)
            drag_factor_d = 0.75 * Cd * rho_c / (rho_d * dp) * ngs.Norm(wd - wc)
            drag_factor_c = -0.75 * Cd * Ad / (Ac * dp) * ngs.Norm(wd - wc)
            a += (dt * drag_factor_d * (ud - uc) * vd) * ngs.dx
            a += (dt * drag_factor_c * (ud - uc) * vc) * ngs.dx

        # — Virtual mass: a_dt (NOT dt-scaled) + a (dt-scaled)
        if self.VM_switch:
            C_VM = self.C_VM[ts]
            VM_d = rho_c * C_VM / rho_d           # phase-d coefficient (positive)
            VM_c = -Ad / Ac * C_VM                # phase-c coefficient (negative)

            # Spatial VM terms (dt-scaled), phase d
            a += (dt * VM_d * (-ngs.InnerProduct(ngs.grad(vd), ngs.OuterProduct(ud, wd))
                               + ngs.InnerProduct(ngs.grad(vd), ngs.OuterProduct(uc, wc))
                               + (-ngs.div(wd)*ud*vd + ngs.div(wc)*uc*vd))) * ngs.dx
            a += (dt * VM_d * ngs.InnerProduct(ngs.OuterProduct(jump(vd), n),
                                                self._NF_UDS_mom(ud, wd))) * ngs.dx(skeleton=True)
            a += (dt * -VM_d * ngs.InnerProduct(ngs.OuterProduct(jump(vd), n),
                                                  self._NF_UDS_mom(uc, wc))) * ngs.dx(skeleton=True)
            if ud_d_reg:
                a += (dt * VM_d * ngs.InnerProduct(ngs.OuterProduct(vd, n),
                                                    self._NF_UDS_mom(ud, wd, 'Dirichlet', True))) \
                     * self._ds(ud_d_reg)
            if uc_d_reg:
                a += (dt * -VM_d * ngs.InnerProduct(ngs.OuterProduct(vd, n),
                                                     self._NF_UDS_mom(uc, wc, 'Dirichlet', True))) \
                     * self._ds(uc_d_reg)
            if ud_n_reg:
                a += (dt * VM_d * ngs.InnerProduct(ngs.OuterProduct(vd, n),
                                                    self._NF_UDS_mom(ud, wd, 'Neumann', True))) \
                     * self._ds(ud_n_reg)
            if uc_n_reg:
                a += (dt * -VM_d * ngs.InnerProduct(ngs.OuterProduct(vd, n),
                                                     self._NF_UDS_mom(uc, wc, 'Neumann', True))) \
                     * self._ds(uc_n_reg)

            # Spatial VM terms (dt-scaled), phase c
            a += (dt * VM_c * (-ngs.InnerProduct(ngs.grad(vc), ngs.OuterProduct(ud, wd))
                               + ngs.InnerProduct(ngs.grad(vc), ngs.OuterProduct(uc, wc))
                               + (-ngs.div(wd)*ud*vc + ngs.div(wc)*uc*vc))) * ngs.dx
            grad_VM_c = C_VM * ngs.grad(Ac) / (Ac**2)
            vm_flux_difference = (ngs.OuterProduct(ud, wd)
                                  - ngs.OuterProduct(uc, wc))
            a += (dt * -(vm_flux_difference.trans * grad_VM_c) * vc) * ngs.dx
            a += (dt * VM_c * ngs.InnerProduct(ngs.OuterProduct(jump(vc), n),
                                                self._NF_UDS_mom(ud, wd))) * ngs.dx(skeleton=True)
            a += (dt * -VM_c * ngs.InnerProduct(ngs.OuterProduct(jump(vc), n),
                                                  self._NF_UDS_mom(uc, wc))) * ngs.dx(skeleton=True)
            if ud_d_reg:
                a += (dt * VM_c * ngs.InnerProduct(ngs.OuterProduct(vc, n),
                                                    self._NF_UDS_mom(ud, wd, 'Dirichlet', True))) \
                     * self._ds(ud_d_reg)
            if uc_d_reg:
                a += (dt * -VM_c * ngs.InnerProduct(ngs.OuterProduct(vc, n),
                                                     self._NF_UDS_mom(uc, wc, 'Dirichlet', True))) \
                     * self._ds(uc_d_reg)
            if ud_n_reg:
                a += (dt * VM_c * ngs.InnerProduct(ngs.OuterProduct(vc, n),
                                                    self._NF_UDS_mom(ud, wd, 'Neumann', True))) \
                     * self._ds(ud_n_reg)
            if uc_n_reg:
                a += (dt * -VM_c * ngs.InnerProduct(ngs.OuterProduct(vc, n),
                                                     self._NF_UDS_mom(uc, wc, 'Neumann', True))) \
                     * self._ds(uc_n_reg)

        # — Lift (dt-scaled)
        if self.Lift_switch:
            Cl = self._get_lift_coeff(wd, wc, ts)
            if self.mesh.dim == 2:
                curl_wc = ngs.grad(wc)[1] - ngs.grad(wc)[2]
                lift_mom_d = Cl * rho_c/rho_d * curl_wc * (
                    (ud[1] - uc[1]) * vd[0] - (ud[0] - uc[0]) * vd[1])
                lift_mom_c = Cl * (-Ad)/(Ac + 1e-30) * curl_wc * (
                    (ud[1] - uc[1]) * vc[0] - (ud[0] - uc[0]) * vc[1])
            else:
                lift_mom_d = Cl * rho_c/rho_d * (ngs.Cross(ud - uc, curl_3d(wc)) * vd)
                lift_mom_c = Cl * (-Ad)/(Ac + 1e-30) * (ngs.Cross(ud - uc, curl_3d(wc)) * vc)
            a += (dt * lift_mom_d) * ngs.dx
            a += (dt * lift_mom_c) * ngs.dx

        return [a]

    def construct_bilinear_time_coefficient(self,
                                             U: List[ProxyFunction],
                                             V: List[ProxyFunction],
                                             dt: Parameter,
                                             time_step: int) -> List:
        return [ngs.CoefficientFunction(0.0) * ngs.dx]

    # ------------------------------------------------------------------
    # Phase 2 — linear form (source and BC terms)
    # ------------------------------------------------------------------

    def construct_linear(self,
                          V: List[ProxyFunction],
                          gfu_0: Optional[List[GridFunction]],
                          dt: Parameter,
                          time_step: int) -> List:
        ts   = time_step
        n    = self.n
        pen  = self.penalty_interior
        comp = self.model_components

        vc, vd = V[comp['u_c']], V[comp['u_d']]
        q, r   = V[comp['p']], V[comp['alpha_c']]

        # Picard iterate (for nonlinear coefficient terms)
        wc = self.UIter.components[comp['u_c']]
        wd = self.UIter.components[comp['u_d']]
        Ac = self.UIter.components[comp['alpha_c']]
        Ad = 1 - Ac

        # Physical parameters
        rho_c = self.rho_c[ts]; rho_d = self.rho_d[ts]
        nu_c  = self.nu_c[ts];  nu_d  = self.nu_d[ts]
        dp    = self.dp[ts]

        # BC regex
        ac_d_reg  = self.dirichlet_names.get('alpha_c', '')
        ac_zg_reg = self._bc_regex('zero_gradient', 'alpha_c')
        ac_zb_reg = self._bc_regex('zero_backflow', 'alpha_c')
        ac_n_reg  = '|'.join(reg for reg in (ac_zg_reg, ac_zb_reg) if reg)
        uc_d_reg  = self.dirichlet_names.get('u_c', '')
        uc_n_reg  = self._bc_regex('zero_stress', 'u_c')
        ud_d_reg  = self.dirichlet_names.get('u_d', '')
        ud_n_reg  = self._bc_regex('zero_stress', 'u_d')

        L = ngs.CoefficientFunction(0.0) * ngs.dx

        # Optional manufactured/general volume sources.  Their signs follow the
        # four strong equations represented by (u_c, u_d, p, alpha_c).
        if 'u_c' in self.f:
            L += (dt * self.f['u_c'][ts] * vc) * ngs.dx
        if 'u_d' in self.f:
            L += (dt * self.f['u_d'][ts] * vd) * ngs.dx
        if 'p' in self.f:
            L += (self.f['p'][ts] * q) * ngs.dx
        if 'alpha_c' in self.f:
            L += (dt * self.f['alpha_c'][ts] * r) * ngs.dx

        # ============================================================
        # 1. Alpha_c — linear contributions
        # ============================================================
        # Explicit (lagged) dispersed-phase compressibility term: the bilinear
        # form drops -div(u_d)*r; it is carried here on the Picard wind w_d.
        L += (dt * ngs.div(wd) * r) * ngs.dx
        for marker, val_list in self.BC.get('dirichlet', {}).get('alpha_c', {}).items():
            val = val_list[ts]
            L += (dt * -r * self._NF_UDS_mass(val, wd, 'Dirichlet', False)) * self._ds(marker)
        if ac_zb_reg:
            # Backflow enters as pure continuous phase (alpha_c = 1).
            L += (dt * -r * Min(wd * n, ngs.CoefficientFunction(0.0))) * self._ds(ac_zb_reg)
        # Artificial diffusion — Dirichlet boundary (linear part).
        if self.diffusion_switch:
            art = self.D_art[ts]
            for marker, val_list in self.BC.get('dirichlet', {}).get('alpha_c', {}).items():
                val = val_list[ts]
                L += (dt * art * pen * val * r) * self._ds(marker)
                L += (-dt * art * val * (ngs.grad(r) * n)) * self._ds(marker)

        # ============================================================
        # 2 & 3. Phase momentum — linear terms (gravity + BCs)
        # ============================================================
        for (v_ts, w_pi, A_pi, rho, nu_lam, d_var, phase) in [
            (vc, wc, Ac, rho_c, nu_c, 'u_c', 'c'),
            (vd, wd, Ad, rho_d, nu_d, 'u_d', 'd'),
        ]:
            # Body force
            L += (dt * self.gravity * v_ts) * ngs.dx

            # Viscous parameters (needed for Nitsche BC terms)
            do_viscous = (self.canonical_form == 'B-TFM' and phase == 'c') or \
                         (self.canonical_form == 'C-TFM')
            if do_viscous:
                nu_eff = ngs.CoefficientFunction(nu_lam)
                factor = nu_eff / A_pi if self.canonical_form == 'B-TFM' else ngs.CoefficientFunction(nu_eff)

            # Convective Dirichlet BC (linear part: outflow from prescribed BC)
            for marker, u_bc_list in self.BC.get('dirichlet', {}).get(d_var, {}).items():
                u_bc = u_bc_list[ts]
                L += (dt * -ngs.InnerProduct(ngs.OuterProduct(v_ts, n),
                                              self._NF_UDS_mom(u_bc, w_pi, 'Dirichlet', False))) \
                     * self._ds(marker)
                # Nitsche viscous Dirichlet (linear part)
                if do_viscous:
                    L += (dt * factor * pen
                          * ngs.InnerProduct(ngs.OuterProduct(u_bc, n),
                                              ngs.OuterProduct(v_ts, n))) * self._ds(marker)
                    L += (dt * -factor * ngs.InnerProduct(
                        ngs.OuterProduct(u_bc, n), ngs.grad(v_ts))) * self._ds(marker)
                    L += (dt * -factor * ngs.InnerProduct(
                        ngs.OuterProduct(u_bc, n), ngs.grad(v_ts).trans)) * self._ds(marker)

            # ZERO_STRESS contributes no data by definition.

        # ============================================================
        # 4. Virtual mass — l_dt (NOT dt-scaled, uses UOld) + l (dt-scaled)
        # ============================================================
        if self.VM_switch:
            C_VM = self.C_VM[ts]
            VM_d = rho_c * C_VM / rho_d
            VM_c = -Ad / Ac * C_VM

            # l (dt-scaled): convective Dirichlet BC contributions
            for marker, u_bc_list in self.BC.get('dirichlet', {}).get('u_d', {}).items():
                u_bc = u_bc_list[ts]
                L += (dt * -VM_d * ngs.InnerProduct(ngs.OuterProduct(vd, n),
                                                     self._NF_UDS_mom(u_bc, wd, 'Dirichlet', False))) \
                     * self._ds(marker)
                L += (dt * -VM_c * ngs.InnerProduct(ngs.OuterProduct(vc, n),
                                                     self._NF_UDS_mom(u_bc, wd, 'Dirichlet', False))) \
                     * self._ds(marker)
            for marker, u_bc_list in self.BC.get('dirichlet', {}).get('u_c', {}).items():
                u_bc = u_bc_list[ts]
                L += (dt * VM_d * ngs.InnerProduct(ngs.OuterProduct(vd, n),
                                                    self._NF_UDS_mom(u_bc, wc, 'Dirichlet', False))) \
                     * self._ds(marker)
                L += (dt * VM_c * ngs.InnerProduct(ngs.OuterProduct(vc, n),
                                                    self._NF_UDS_mom(u_bc, wc, 'Dirichlet', False))) \
                     * self._ds(marker)

        # ============================================================
        # 5. Laminar dispersion (linear source term)
        # ============================================================
        if self.Disp_switch and self.drag_switch:
            Cd   = self._get_drag_coeff(wd, wc, Ad, ts)
            H    = 1 - 1.166*(1 - Ac) + 0.5*(1 - Ac)**2
            Cdis = self.Cdis[ts]
            disp_d = 0.75*Cd*Cdis*rho_c/rho_d * H * ngs.Norm(wd - wc)**2
            disp_c = -0.75*Cd*Cdis*Ad/(Ac + 1e-30) * H * ngs.Norm(wd - wc)**2
            L += (dt * disp_d * ngs.grad(Ac) * vd) * ngs.dx
            L += (dt * disp_c * ngs.grad(Ac) * vc) * ngs.dx

        # ============================================================
        # 6. Interior injection source (on the 'injection' material region)
        # ============================================================
        if self.injection_switch and self.inj_mass_flowrate:
            mdot = self.inj_mass_flowrate
            dx_inj = ngs.dx(definedon=self.mesh.Materials(self.injection_region))
            L += (dt * -mdot * r) * dx_inj          # dispersed-phase mass
            L += (dt *  mdot * q) * dx_inj          # mixture mass
            if self.mesh.dim == 2:
                inj_vec = ngs.CoefficientFunction((0, self.inj_velocity))
            else:
                inj_vec = ngs.CoefficientFunction((0, self.inj_velocity, 0))
            L += (dt * mdot * inj_vec * vc) * dx_inj  # continuous-phase momentum

        return [L]

    def construct_imex_explicit(self,
                                 V: List[ProxyFunction],
                                 gfu_0: Optional[List[GridFunction]],
                                 dt: Parameter,
                                 time_step: int) -> List:
        return [ngs.CoefficientFunction(0.0) * ngs.dx]

    # ------------------------------------------------------------------
    # Phase 3 — nonlinear (Picard) solver
    # ------------------------------------------------------------------

    def solve_single_step(self,
                           a_lst: List[BilinearForm],
                           L_lst: List[LinearForm],
                           precond_lst: List[Preconditioner],
                           gfu: GridFunction,
                           time_step: int = 0) -> bool:
        # Cold-start guard: the adaptive solvers hand us fresh GridFunctions
        # (gfu_long/gfu_short) on the first step with only the velocity Dirichlet
        # BCs applied -- alpha_c is still zero. A zero alpha_c is a degenerate
        # linearization wind for this model (drag /Ac -> /0), so seed it from the
        # initial condition before it is used as UIter.  (Keys on alpha_c since
        # velocity BCs leave it untouched.)
        comp = self.model_components
        if gfu.components[comp['alpha_c']].vec.Norm() == 0.0:
            gfu.vec.data = self.IC.vec

        gfu_prev = ngs.GridFunction(self.fes)

        for _it in range(self.nonlinear_max_iters):
            gfu_prev.vec.data = gfu.vec

            # Update Picard wind so that Assemble() picks up the new values
            self.UIter.vec.data = gfu.vec

            self.apply_dirichlet_bcs_to(gfu, time_step)

            a_lst[0].Assemble()
            L_lst[0].Assemble()
            if precond_lst[0] is not None:
                precond_lst[0].Update()

            self.linear_solve(a_lst[0], L_lst[0], precond_lst[0], gfu)

            # Per-component relaxation
            for i, rf in enumerate(self.relax_factors):
                if rf < 1.0:
                    gfu.components[i].vec.data = (
                        rf * gfu.components[i].vec
                        + (1.0 - rf) * gfu_prev.components[i].vec)

            # Bound-preserving Bezier/Bernstein limiter: GUARANTEES the per-element
            # polynomial stays in bounds everywhere. Order is the
            # scalar L2 order used for alpha_c in _construct_fes.
            if self.slope_limiter:
                ord_s = max(self.interp_ord - 1, 0)
                ac = gfu.components[comp['alpha_c']]
                self._lim.bezier_bound(ac, ac.space, ord_s, (0.0, 1.0))

            # Match INS convergence testing: evaluate each local-error component
            # independently and exclude pressure.
            converged = True
            for name, include in self.model_local_error_components.items():
                if not include:
                    continue
                i = comp[name]
                fes_component = self.fes.components[i]
                err = norm('l2_norm', gfu_prev.components[i], gfu.components[i],
                           self.mesh, fes_component, average=False)
                solution_norm = mean(gfu.components[i], self.mesh)
                tolerance = (self.abs_nonlinear_tolerance
                             + self.rel_nonlinear_tolerance * solution_norm)
                if err >= tolerance:
                    converged = False
                    break

            if converged:
                logging.info(f'TFM Picard converged in {_it + 1} iteration(s).')
                return True

        logging.warning('TFM Picard did NOT converge within '
                        f'{self.nonlinear_max_iters} iterations.')
        return False

    def update_linearization(self, gfu: GridFunction) -> None:
        self.UIter.vec.data = gfu.vec

    def linearized_solve(self, a_assembled: BilinearForm, L_assembled: LinearForm,
                         precond: Preconditioner, gfu: GridFunction):
        """Perform one stationary Picard iteration and report its change."""
        previous = ngs.GridFunction(self.fes)
        previous.vec.data = gfu.vec
        self.linear_solve(a_assembled, L_assembled, precond, gfu)

        # The stationary solver performs one Picard solve per outer iteration,
        # so apply the same component-wise under-relaxation used by
        # solve_single_step for transient TFM solves.
        for i, relaxation in enumerate(self.relax_factors):
            if relaxation < 1.0:
                gfu.components[i].vec.data = (
                    relaxation * gfu.components[i].vec
                    + (1.0 - relaxation) * previous.components[i].vec)

        error = 0.0
        solution_norm = 0.0
        for name, include in self.model_local_error_components.items():
            if not include:
                continue
            component = self.model_components[name]
            space = self.fes.components[component]
            error = max(error, norm('l2_norm', previous.components[component],
                                    gfu.components[component], self.mesh, space,
                                    average=False))
            solution_norm = max(solution_norm, mean(gfu.components[component], self.mesh))
        return error, solution_norm
