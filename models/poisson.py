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
from helpers.ngsolve_ import get_special_functions
from helpers.dg import jump, grad_avg
from config_functions import ConfigParser
from models import Model
from typing import Tuple, List, Union
from ngsolve.comp import FESpace, ProxyFunction, GridFunction
from ngsolve import Parameter


class Poisson(Model):
    """
    A poisson model.
    """

    def __init__(self, config: ConfigParser, t_param: ngs.Parameter) -> None:
        # Specify information about the model components
        # NOTE: These MUST be set before calling super(), since it's needed in superclass' __init__
        self.model_components               = {'u': None}
        self.model_local_error_components   = {'u': True}
        self.time_derivative_components     = {'u': True}

        # Pre-define which BCs are accepted for this model, all others are thrown out.
        self.BC_init = {'robin':     {},
                        'dirichlet': {},
                        'neumann':   {},
                        'pinned':    {}}

        super().__init__(config, t_param)

    @staticmethod
    def allows_explicit_schemes() -> bool:
        return True

    def _construct_fes(self) -> FESpace:
        return getattr(ngs, self.element['u'])(self.mesh, order=self.interp_ord,
                                             dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)

    def _set_model_parameters(self) -> None:
        self.dc = self.model_functions.model_parameters_dict['diffusion_coefficient']['all']
        self.f = self.model_functions.model_functions_dict['source']['all']

    def construct_bilinear(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter = ngs.Parameter(1.0),
                           explicit_bilinear: bool = False) -> ngs.BilinearForm:
        # Split up the trial (weighting) and test functions
        u = U[0]
        v = V[0]

        if self.DIM:
            # Use the diffuse interface method.
            a = self._bilinear_DIM(u, v, dt, explicit_bilinear)
        else:
            # Solve on a standard conformal mesh.
            a = self._bilinear_no_DIM(u, v, dt, explicit_bilinear)

        return a

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Union[GridFunction, None] = None,
                         dt: Parameter = ngs.Parameter(1.0)) -> ngs.LinearForm:
        # V is a list of length 1 for Poisson, get just the first element
        v = V[0]

        if self.DIM:
            # Use the diffuse interface method.
            L = self._linear_DIM(v, dt)
        else:
            # Solve on a standard conformal mesh.
            L = self._linear_no_DIM(v, dt)

        return L

    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        u = self.fes.TrialFunction()
        v = self.fes.TestFunction()

        return [u], [v]

    def single_iteration(self, a: ngs.BilinearForm, L: ngs.LinearForm, precond: ngs.Preconditioner,
                         gfu: ngs.GridFunction) -> None:

        self.construct_and_run_solver(a, L, precond, gfu)

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################

    def _bilinear_DIM(self, u: ProxyFunction, v: ProxyFunction, dt: Parameter, explicit_bilinear: bool) -> ngs.BilinearForm:
        """ Bilinear form when the diffuse interface method is being used. Handles both CG and DG. """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        # Laplacian term
        a = dt * self.dc * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u to zero where phi is zero.
        # Applying a penalty can help convergence, but the penalty seems to interfere with Neumann BCs.
        if self.DIM_BC.get('neumann', {}).get('u', {}):
            a += dt * u * v * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx
        else:
            a += dt * alpha * u * v * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += dt * self.dc * (
                    - jump_u * n * avg_grad_v  # U
                    - avg_grad_u * jump_v * n  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                    + alpha * jump_u * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                    ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += dt * self.dc * (
                    - u * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    - ngs.Grad(u) * n * v  # ∇u^ = ∇u
                    + alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += dt * self.dc * (
                ngs.Grad(u) * self.DIM_solver.grad_phi_gfu * v
                + ngs.Grad(v) * self.DIM_solver.grad_phi_gfu * u
                + alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # DIM Robin BCs for u.
        for marker in self.DIM_BC.get('robin', {}).get('u', {}):
            r, q = self.DIM_BC['robin']['u'][marker]
            a += dt * self.dc * (
                r * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker]
            a += -dt * self.dc * r * u * v * self._ds(marker)

        return a

    def _bilinear_no_DIM(self, u: ProxyFunction, v: ProxyFunction, dt: Parameter, explicit_bilinear: bool) -> ngs.BilinearForm:
        """ Bilinear form when the diffuse interface method is not being used. Handles both CG and DG. """

        # Laplacian term
        a = dt * self.dc * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx

        # Bulk of Bilinear form
        if self.DG:
            # Define the special DG functions.
            n, _, alpha = get_special_functions(self.mesh, self.nu)

            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += dt * self.dc * (
                    - jump_u * n * avg_grad_v  # U
                    - avg_grad_u * n * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                    + alpha * jump_u * jump_v  # 1/2 term for u+=u- on 𝚪_I from ∇u^
                    ) * ngs.dx(skeleton=True)

            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for Dirichlet BCs
                a += dt * self.dc * (
                    - u * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    - ngs.Grad(u) * n * v  # ∇u^ = ∇u
                    + alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker]
            a += -dt * self.dc * r * u * v * self._ds(marker)

        return a

    def _linear_DIM(self, v: ProxyFunction, dt: Parameter) -> ngs.LinearForm:
        """ Linear form when the diffuse interface method is being used. Handles both CG and DG. """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        # Source term
        L = dt * self.f * v * self.DIM_solver.phi_gfu * ngs.dx

        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                # Penalty terms for conformal Dirichlet BCs
                g = self.BC['dirichlet']['u'][marker]
                L += dt * self.dc * (
                    alpha * g * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - g * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    ) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            # Penalty terms for DIM Dirichlet BCs
            g = self.DIM_BC['dirichlet']['u'][marker]
            L += dt * self.dc * (
                ngs.Grad(v) * self.DIM_solver.grad_phi_gfu * g
                + alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # DIM Neumann BCs for u.
        for marker in self.DIM_BC.get('neumann', {}).get('u', {}):
            h = self.DIM_BC['neumann']['u'][marker]
            L += dt * self.dc * (-h * v * self.DIM_solver.mag_grad_phi_gfu
                            ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # DIM Robin BCs for u.
        for marker in self.DIM_BC.get('robin', {}).get('u', {}):
            r, q = self.DIM_BC['robin']['u'][marker]
            L += dt * self.dc * (r * q * v * self.DIM_solver.mag_grad_phi_gfu
                            ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Neumann BCs for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker]
            L += dt * self.dc * h * v * self._ds(marker)

        # Conformal Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker]
            L += -dt * self.dc * r * q * v * self._ds(marker)

        return L

    def _linear_no_DIM(self, v: ProxyFunction, dt: Parameter) -> ngs.LinearForm:
        """ Linear form when the diffuse interface method is not being used. Handles both CG and DG. """

        # Source term
        L = dt * self.f * v * ngs.dx

        # Dirichlet BCs for u, they only get added if using DG
        if self.DG:
            # Define the special DG functions.
            n, _, alpha = get_special_functions(self.mesh, self.nu)

            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += dt * self.dc * (
                    alpha * g * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    - g * n * ngs.Grad(v)  # 1/2 of penalty for u=g on 𝚪_D
                    ) * self._ds(marker)

        # Neumann BCs for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker]
            L += dt * self.dc * h * v * self._ds(marker)

        # Robin BCs for u
        for marker in self.BC.get('robin', {}).get('u', {}):
            r, q = self.BC['robin']['u'][marker]
            L += -dt * self.dc * r * q * v * self._ds(marker)

        return L