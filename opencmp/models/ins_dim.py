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
from typing import List, Optional, Union

from ..helpers.ngsolve_ import get_special_functions
from ..helpers.dg import avg, grad_avg, jump
from . import INS

from ngsolve import BilinearForm, Grad, GridFunction, IfPos, InnerProduct, LinearForm, Norm, OuterProduct, Parameter, \
    div, dx
from ngsolve.comp import ProxyFunction


class INSDIM(INS):
    """
    A single phase incompressible Navier-Stokes model with the Diffuse Interface Method.
    """

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        w = self._get_wind(U, time_step)

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test function for velocity
        u = U[self.model_components['u']]
        v = V[self.model_components['u']]

        # Domain integrals.
        a = dt * (
                self.kv[time_step] * InnerProduct(Grad(u), Grad(v))  # Stress, Newtonian
        ) * self.DIM_solver.phi_gfu * dx

        if self.linearize == 'Oseen':
            # Linearized convection term.
            a += -dt * InnerProduct(OuterProduct(u, w), Grad(v)) * self.DIM_solver.phi_gfu * dx

        # Force u to zero where phi is zero.
        # If DIM rigid body motion is being used force u to the velocity of the rigid body instead.
        a += dt * alpha * u * v * (1.0 - self.DIM_solver.phi_gfu) * dx

        if self.DG:
            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += dt * (
                        self.kv[time_step] * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                        - self.kv[time_step] * InnerProduct(Grad(u), OuterProduct(v, n))  # ∇u^ = ∇u
                        - self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(u, n))  # 1/2 of penalty for u=g on 𝚪_D
                ) * self._ds(self.dirichlet_names['u'])

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection term)
                    a += dt * v * (0.5 * w * n * u + 0.5 * Norm(w * n) * u) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += dt * (
                    self.kv[time_step] * InnerProduct(Grad(u), OuterProduct(v, self.DIM_solver.grad_phi_gfu))
                    + self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(u, self.DIM_solver.grad_phi_gfu))
                    + self.kv[time_step] * alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
            ) * self.DIM_solver.mask_gfu_dict[marker] * dx

            if self.linearize == 'Oseen':
                # Additional convection term penalty.
                a += dt * (v * (
                        0.5 * w * -self.DIM_solver.grad_phi_gfu * u + 0.5
                        * Norm(w * -self.DIM_solver.grad_phi_gfu) * u
                )) * self.DIM_solver.mask_gfu_dict[marker] * dx

        if self.linearize == 'Oseen':
            # Conformal stress BC needs a no-backflow component in the bilinear form.
            for marker in self.BC.get('stress', {}).get('u', {}):
                if self.DG:
                    a += dt * v * (IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
                else:
                    a += dt * v.Trace() * (IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Conformal parallel flow BC
        for marker in self.BC.get('parallel', {}).get('u', {}):
            if self.DG:
                a += dt * v * (u - n * InnerProduct(u, n)) * self._ds(marker)
            else:
                a += dt * v.Trace() * (u.Trace() - n * InnerProduct(u.Trace(), n)) * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return [a]

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:

        w = self._get_wind(U, time_step)

        # Define the special DG functions.
        n, _, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the trial and test functions for velocity and pressure.
        u, p = U[self.model_components['u']], U[self.model_components['p']]
        v, q = V[self.model_components['u']], V[self.model_components['p']]

        # Domain integrals.
        a = dt * (
                - div(u) * q     # Conservation of mass
                - div(v) * p     # Pressure
                - 1e-10 * p * q  # Stabilization term
        ) * self.DIM_solver.phi_gfu * dx

        # Force grad(p) to zero where phi is zero. If rigid body motion is being used ignore this.
        if not self.DIM_solver.rigid_body_motion:
            a += -dt * p * div(v) * (1.0 - self.DIM_solver.phi_gfu) * dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(v)
            avg_grad_v = grad_avg(v)

            # Penalty for discontinuities
            a += dt * (
                    self.kv[time_step] * alpha * InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I from ∇u^
                    - self.kv[time_step] * InnerProduct(avg_grad_u, OuterProduct(jump_v, n))  # Stress
                    - self.kv[time_step] * InnerProduct(avg_grad_v, OuterProduct(jump_u, n))  # U
            ) * self.DIM_solver.phi_gfu * dx(skeleton=True)

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                a += dt * jump_v * (w * n * avg_u + 0.5 * Norm(w * n) * jump_u) * self.DIM_solver.phi_gfu * dx(skeleton=True)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:

        w = self._get_wind(gfu_0, time_step)

        # Define the special DG functions.
        n, h, alpha, I_mat = get_special_functions(self.mesh, self.nu)

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Domain integrals.
        L = dt * v * self.f['u'][time_step] * self.DIM_solver.phi_gfu * dx

        # Force u to zero where phi is zero. If DIM rigid body motion is being used force u to the velocity of the
        # rigid body instead.
        if self.DIM_solver.rigid_body_motion:
            for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
                g = self.DIM_BC['dirichlet']['u'][marker][time_step]
                L += dt * alpha * g * v * (1.0 - self.DIM_solver.phi_gfu) * self.DIM_solver.mask_gfu_dict[marker] * dx

        if self.DG:
            # Conformal Dirichlet BCs for u.
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker][time_step]
                L += dt * (
                        self.kv[time_step] * alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                        - self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(g, n))  # 1/2 of penalty for u=g
                ) * self._ds(marker)

                if self.linearize == 'Oseen':
                    # Additional 1/2 of uw^ (convection)
                    L += dt * v * (-0.5 * w * n * g + 0.5 * Norm(w * n) * g) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            g = self.DIM_BC['dirichlet']['u'][marker][time_step]
            L += dt * (
                    self.kv[time_step] * InnerProduct(Grad(v), OuterProduct(g, self.DIM_solver.grad_phi_gfu))
                    + self.kv[time_step] * alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
            ) * self.DIM_solver.mask_gfu_dict[marker] * dx

            if self.linearize == 'Oseen':
                # Additional penalty for the convection term.
                L += dt * v * (
                        0.5 * w * self.DIM_solver.grad_phi_gfu * g
                        + 0.5 * Norm(w * -self.DIM_solver.grad_phi_gfu) * g
                ) * self.DIM_solver.mask_gfu_dict[marker] * dx

        # Conformal stress BC
        for marker in self.BC.get('stress', {}).get('u', {}):
            h = self.BC['stress']['u'][marker][time_step]
            if self.DG:
                L += dt * v * h * self._ds(marker)
            else:
                L += dt * v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:

        # Separate out the test function for velocity.
        v = V[self.model_components['u']]

        # Velocity linearization term
        gfu_u = gfu_0[self.model_components['u']]

        # Linearized convection term.
        L = -dt * InnerProduct(Grad(gfu_u) * gfu_u, v) * self.DIM_solver.phi_gfu * dx

        return [L]
