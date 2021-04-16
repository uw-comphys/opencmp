"""
Copyright 2021 the authors (see AUTHORS file for full list)

This file is part of OpenCMP.

OpenCMP is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

OpenCMP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with OpenCMP.  If not, see <https://www.gnu.org/licenses/>.
"""

import ngsolve as ngs
from helpers.ngsolve_ import construct_p_mat, get_special_functions
from helpers.dg import jump, grad_avg
from models import Model
from ngsolve.comp import FESpace, ProxyFunction, GridFunction
from ngsolve import Parameter
from typing import Tuple, List, Union
from config_functions import ConfigParser


class Stokes(Model):
    """
    A single phase Stokes model.
    """
    def __init__(self, config: ConfigParser, t_param: ngs.Parameter) -> None:
        # Specify information about the model components
        # NOTE: These MUST be set before calling super(), since it's needed in superclass' __init__
        self.model_components               = {'u': 0,      'p': 1}
        self.model_local_error_components   = {'u': True,   'p': True}
        self.time_derivative_components     = {'u': True,   'p': False}

        # Pre-define which BCs are accepted for this model, all others are thrown out.
        self.BC_init = {'dirichlet':    {},
                        'stress':       {},
                        'pinned':       {}}

        super().__init__(config, t_param)

    @staticmethod
    def allows_explicit_schemes() -> bool:
        # Stokes cannot work with explicit schemes
        return False

    def _set_model_parameters(self) -> None:
        self.kv = self.model_functions.model_parameters_dict['kinematic_viscosity']['all']
        self.f = self.model_functions.model_functions_dict['source']['all']

    def _construct_fes(self) -> FESpace:
        if not self.DG:
            if self.element['u'] == 'HDiv' or self.element['u'] == 'RT':
                print('We recommended that you NOT use HDIV spaces without DG due to numerical issues.')
            if self.element['p'] == 'L2':
                print('We recommended that you NOT use L2 spaces without DG due to numerical issues.')

        if self.element['u'] == 'RT':
            # Raviart-Thomas elements are a type of HDiv finite element.
            fes_u = ngs.HDiv(self.mesh, order=self.interp_ord, dirichlet=self.dirichlet_names.get('u', ''),
                             dgjumps=self.DG, RT=True)
        else:
            fes_u = getattr(ngs, self.element['u'])(self.mesh, order=self.interp_ord,
                                              dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)

        if self.element['p'] == 'L2' and 'p' in self.dirichlet_names.keys():
            raise ValueError('Not able to pin pressure at a point on L2 spaces.')
        else:
            fes_p = getattr(ngs, self.element['p'])(self.mesh, order=self.interp_ord - 1,
                                                  dirichlet=self.dirichlet_names.get('p', ''), dgjumps=self.DG)

        return ngs.FESpace([fes_u, fes_p], dgjumps=self.DG)

    def construct_bilinear(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter = ngs.Parameter(1.0),
                           explicit_bilinear: bool = False) -> ngs.BilinearForm:
        # Split up the trial (weighting) and test functions
        u, p = U
        v, q = V

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_DIM(u, p, v, q, dt, explicit_bilinear)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_no_DIM(u, p, v, q, dt, explicit_bilinear)

        return a

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Union[GridFunction, None] = None,
                         dt: Parameter = ngs.Parameter(1.0)) -> ngs.LinearForm:
        # Split up the test functions
        v, q = V

        if self.DIM:
            # Use the diffuse interface method.
            L = self._linear_DIM(v, dt)
        else:
            # Solve on a standard conformal mesh.
            L = self._Linear_no_DIM(v, dt)

        return L

    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        # Define the test and trial functions.
        u, p = self.fes.TrialFunction()
        v, q = self.fes.TestFunction()

        return [u, p], [v, q]

    def single_iteration(self, a: ngs.BilinearForm, L: ngs.LinearForm, precond: ngs.Preconditioner,
                         gfu: ngs.GridFunction) -> None:

        self.construct_and_run_solver(a, L, precond, gfu)

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################

    def _bilinear_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction, dt: Parameter,
                         explicit_bilinear: bool) -> ngs.BilinearForm:
        """ Bilinear form when the diffuse interface method is being used. Handles both CG and DG. """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        a = dt * (
            self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            #- 1e-10 * p * q   # Stabilization term.
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u and grad(p) to zero where phi is zero.
        a += dt * (
            alpha * u * v
            - p * ngs.div(v)
            ) * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += -dt * self.kv * (
                    ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                    + ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                    - alpha * ngs.InnerProduct(jump_u, jump_v)  # Term for u+=u- on 𝚪_I from ∇u^
                    ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += -dt * self.kv * (
                    ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                    - alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += dt * self.kv * (
                ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, self.DIM_solver.grad_phi_gfu))
                + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, self.DIM_solver.grad_phi_gfu))
                + alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # TODO: Add non-Dirichlet DIM BCs.

        return a

    def _bilinear_no_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction, dt: Parameter,
                         explicit_bilinear: bool) -> ngs.BilinearForm:
        """ Bilinear form when the diffuse interface method is not being used. Handles both CG and DG. """

        a = dt * (
            self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            - 1e-10 * p * q   # Stabilization term
            ) * ngs.dx

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        # Bulk of Bilinear form
        if self.DG:
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += -dt * self.kv * (
                    ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                    + ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                    - alpha * ngs.InnerProduct(jump_u, jump_v)  # Term for u+=u- on 𝚪_I from ∇u^
                    ) * ngs.dx(skeleton=True)

            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += -dt * self.kv * (
                    ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                    - alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        return a

    def _linear_DIM(self, v: ProxyFunction, dt: Parameter) -> ngs.LinearForm:
        """ Linear form when the diffuse interface method is being used. Handles both CG and DG. """

        # Define the base linear form
        L = dt * v * self.f * self.DIM_solver.phi_gfu * ngs.dx

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        if self.DG:
            # Conformal Dirichlet BCs for u
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += dt * self.kv * (
                    alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    ) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            # Penalty terms for DIM Dirichlet BCs
            g = self.DIM_BC['dirichlet']['u'][marker]
            L += dt * self.kv * (
                alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
                + ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, self.DIM_solver.grad_phi_gfu))
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Stress BCs
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker]

            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return L

    def _Linear_no_DIM(self, v: ProxyFunction, dt: Parameter) -> ngs.LinearForm:
        """ Linear form when the diffuse interface method is not being used. Handles both CG and DG. """

        # Define the base linear form
        L = dt * v * self.f * ngs.dx

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        if self.DG:
            # Dirichlet BCs for u
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += dt * self.kv * (
                    alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    - ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    ) * self._ds(marker)

        # Stress BCs
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker]

            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        return L
