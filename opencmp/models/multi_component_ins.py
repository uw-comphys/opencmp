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
from typing import Dict, List, Optional, Set, Union

import ngsolve
from ngsolve import BilinearForm, FESpace, Grad, InnerProduct, LinearForm, GridFunction, Preconditioner, \
    ds, dx, Parameter, CoefficientFunction
from ngsolve.comp import ProxyFunction

from ..helpers.dg import grad_avg, jump
from ..helpers.math import Min, Max
from ..helpers.ngsolve_ import get_special_functions
from ..config_functions import ConfigParser
from . import INS

# TODO: These might be a good reference
# https://github.com/NGSolve/modeltemplates/tree/master/templates
# https://github.com/NGSolve/modeltemplates/blob/master/introduction/introduction.ipynb


class MultiComponentINS(INS):
    """
    A single phase multicomponent incompressible Navier-Stokes model.
    """

    def _add_multiple_components(self, config: ConfigParser) -> None:
        """
        Function to add multiple components to the model from the config file.

        Args:
            config: The ConfigParser
        """

        # Name of new components
        self.extra_components                           = config.get_list(['OTHER', 'component_names'], str)
        # Whether each component should be added to the local error calculation
        new_components_in_error_calc: Dict[str, bool]   = config.get_dict(['OTHER', 'component_in_error_calc'], "")
        # Whether each component has a time derivative associated with it
        new_components_in_time_deriv: Dict[str, bool]   = config.get_dict(['OTHER', 'component_in_time_deriv'], "")

        # Ensure that the user specified the required information about each variable
        assert len(self.extra_components) == len(new_components_in_error_calc)
        assert len(self.extra_components) == len(new_components_in_time_deriv)

        # If the velocity is fixed, remove info associated with it from the model
        if (self.fixed_velocity):
            self.model_components = dict()
            self.model_local_error_components = dict()
            self.time_derivative_components = [dict()]

        # Number of default variables
        num_existing = len(self.model_components)
        num_existing_ic = len(self.model_components_ic)

        for i in range(len(self.extra_components)):
            component = self.extra_components[i]

            self.model_components[component]                = i + num_existing
            self.model_components_ic[component]             = i + num_existing_ic
            self.model_local_error_components[component]    = new_components_in_error_calc[component]
            self.time_derivative_components[0][component]   = new_components_in_time_deriv[component]

    def _construct_fes(self) -> FESpace:
        if self.fixed_velocity:
            fes_total = []
        else:
            # Create the FE spaces for velocity and pressure
            fes_total = self._construct_fes_helper()

        # Iterate over each component and add a fes for each
        for component in self.extra_components:
            element_type_for_component = self.element[component]
            kwargs = {'mesh': self.mesh,
                      'order': self.interp_ord,
                      'dgjumps': self.DG}
            if element_type_for_component == "L2":
                if not self.DG:
                    print('We recommended that you NOT use L2 spaces without DG due to numerical issues.')
            else:
                kwargs['dirichlet'] = self.dirichlet_names.get(component, '')

            fes_total.append(
                getattr(ngsolve, element_type_for_component)(**kwargs)
            )

        return FESpace(fes_total, dgjumps=self.DG)

    def _construct_ic_fes(self) -> FESpace:
        # Create the FE spaces for velocity and pressure
        fes_total = self._construct_fes_helper()

        # Iterate over each component and add a fes for each
        for component in self.extra_components:
            fes_total.append(
                getattr(ngsolve, self.element[component])(self.mesh, order=self.interp_ord,
                                                          dirichlet=self.dirichlet_names.get(component, ''),
                                                          dgjumps=self.DG)
            )

        return FESpace(fes_total, dgjumps=self.DG)

    def _define_bc_types(self) -> List[str]:
        # TODO: For James. Neumann BC or diffusive flux BC? Which makes more sense?
        # TODO: If we go with a Neumann BC need to add in the diffusion coefficient in the bilinear form.
        return super()._define_bc_types() + ['neumann', 'total_flux', 'surface_rxn']

    def _post_init(self) -> None:
        if self.DIM:
            raise NotImplementedError('DIM is not yet implemented.')

        if self.DG and self.linearize == 'IMEX':
            raise NotImplementedError('DG IMEX is not yet implemented.')

        if self.linearize == 'IMEX' and self.config.get_item(['TRANSIENT', 'scheme'], str) != 'euler IMEX':
            raise NotImplementedError('Higher order IMEX schemes not yet implemeted for this model')

        if self.fixed_velocity:
            self.W = self._construct_linearization_terms()

        super()._post_init()

    def _pre_init(self) -> None:
        # Read in and add the multiple components
        self._add_multiple_components(self.config)

    def _set_model_parameters(self) -> None:
        # Read in kinematic viscosity and source terms
        super()._set_model_parameters()
        # Read in diffusion coefficient
        self.Ds = self.model_functions.model_parameters_dict['diffusion_coefficients']

        # Ensure that a value was loaded for each extra component
        assert len(self.Ds) == len(self.extra_components)
        assert len(self.f) == len(self.extra_components) + 0 if self.fixed_velocity else 1  # Extra +1 since velocity also has a source term, +0 if velocity is fixed since it's equation is not used

    def _get_wind(self, U, time_step):
        if self.fixed_velocity:
            return self.W[self.model_components_ic['u']]
        else:
            return super()._get_wind(U, time_step)

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:

        if self.fixed_velocity:
            a = CoefficientFunction(0) * dx
        else:
            # Calculate the hydrodynamic contribution to the bilinear terms
            a = super().construct_bilinear_time_coefficient(U, V, dt, time_step)[0]

        if self.linearize == 'Oseen':
            w = self._get_wind(U, time_step)

        # Define the special DG functions.
        n, _, alpha, _ = get_special_functions(self.mesh, self.nu)

        if self.DG:
            for comp in self.extra_components:
                # Trial and test functions
                c = U[self.model_components[comp]]
                r = V[self.model_components[comp]]

                jump_c = jump(c)
                avg_grad_c = grad_avg(c)

                jump_r = jump(r)
                avg_grad_r = grad_avg(r)

                # Diffusion on internal facets
                if self.Ds[comp][time_step] != 0:
                    a += dt * self.Ds[comp][time_step] * (
                        InnerProduct(alpha*jump_c*n - avg_grad_c, jump_r*n)
                        - InnerProduct(avg_grad_r, jump_c * n)
                    ) * dx(skeleton=True)

                # Advection on internal facets
                if self.linearize == 'Oseen':
                    a += dt * jump_r * (
                            c * Max(InnerProduct(w, n), 0)
                            + c.Other() * Min(InnerProduct(w, n), 0)
                    ) * dx(skeleton=True)

        return [a]

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        if self.fixed_velocity:
            a = CoefficientFunction(0) * dx
        else:
            # Calculate the hydrodynamic contribution to the bilinear terms
            a = super().construct_bilinear_time_ODE(U, V, dt, time_step)[0]

        if self.linearize == 'Oseen':
            w = self._get_wind(U, time_step)

        # Define the special DG functions.
        n, _, alpha, _ = get_special_functions(self.mesh, self.nu)

        # Transport of the mixture components.
        for comp in self.extra_components:
            # Trial and test functions
            c = U[self.model_components[comp]]
            r = V[self.model_components[comp]]

            # Diffusion in bulk
            if self.Ds[comp][time_step] > 0:
                a += dt * self.Ds[comp][time_step] * InnerProduct(Grad(c), Grad(r)) * dx

            # NOTE: Reaction terms can go in either the bilinear form or the linear form depending on if the reaction
            #       coefficients are functions of the trial functions. Basic type-checking can't be used since the
            #       reaction coefficients will always be coefficientfunctions or gridfunctions. Instead, check if the
            #       reaction coefficients are gridfunctions (go in linear form) and if they are coefficientfunctions
            #       expand out the coefficientfunction tree and check for the presence of "trial-function" (goes in
            #       bilinear form).

            # Surface reaction
            # TODO: Does not work
            for marker in self.BC.get('surface_rxn', {}).get(comp, {}):
                val = self.BC['surface_rxn'][comp][marker][time_step]
                if isinstance(val, ProxyFunction) or (isinstance(val, CoefficientFunction) and ('trial-function' in val.__str__())):
                    if self.DG:
                        a += -dt * val * r * self._ds(marker)
                    else:
                        a += -dt * val * r.Trace() * self._ds(marker)

            # Bulk reaction
            # TODO: Temp solution to test, implement better once parser has been updated
            val = self.f[comp][time_step]
            if isinstance(val, ProxyFunction) or (isinstance(val, CoefficientFunction) and ('trial-function' in val.__str__())):
                # NOTE: The negative sign is here since the source term was brought over to bilinear side, it usually
                #       does not have one since it is on the linear term side.
                if self.BC.get('surface_rxn', {}).get(comp, None) is not None:
                    marker = list(self.BC['surface_rxn'][comp].keys())[0]
                    if self.DG:
                        a += -dt * val * r * self._ds(marker)
                    else:
                        a += -dt * val * r.Trace() * self._ds(marker)
                else:
                    a += -dt * val * r * dx

            if self.linearize == 'Oseen':
                # Advection in bulk
                a -= dt * c * InnerProduct(w, Grad(r)) * dx

                # 1/2 of Total Flux BC ("Natural" BC), other half in linear term
                #   The upwinding term needs to be added this way since it must be present even if a value for h
                #   is not specified for that boundary.
                #   By removing the neumann, dirichlet, and reaction BCs, all that is left are the total_flux
                #   and "unlabelled" BCs, both of which need the upwinding term.
                total_flux_bc_markers: Set[str] = set(self.mesh.GetBoundaries())
                for _bc_name in ['dirichlet', 'neumann', 'surface_rxn']:
                    total_flux_bc_markers.difference_update(set(self.BC.get(_bc_name, {}).get(comp, {})))
                for marker in total_flux_bc_markers:
                    # Upwinding term, always add.
                    if self.DG:
                        a += dt * r * c * Max(InnerProduct(w, n), 0) * self._ds(marker)
                    else:
                        a += dt * r.Trace() * c * Max(InnerProduct(w, n), 0) * self._ds(marker)

                # 1/2 of the Neumann BC, other half in the linear form
                for marker in self.BC.get('neumann', {}).get(comp, {}):
                    if self.Ds[comp][time_step] == 0:
                        raise ValueError('Trying to apply a neuman boundary condition for '
                                         'purely advective flow (diffusion coefficient is 0).')
                    if self.DG:
                        a += dt * r * c * InnerProduct(w, n) * self._ds(marker)
                    else:
                        a += dt * r.Trace() * c * InnerProduct(w, n) * self._ds(marker)

                if self.DG:
                    # 1/2 of diffusion on Dirichlet BC (1st & 2nd line)
                    # AND
                    # 1/2 of penalty on Dirichlet BC (3rd line)
                    if self.Ds[comp][time_step] != 0:
                        for marker in self.BC.get('dirichlet').get(comp, {}):
                            a += dt * self.Ds[comp][time_step] * (
                                alpha * r * c
                                - r * InnerProduct(Grad(c), n)
                                - c * InnerProduct(Grad(r), n)
                            ) * self._ds(marker)

                    # 1/2 of advection on dirichlet BCs
                    for marker in self.BC.get('dirichlet', {}).get(comp, {}):
                        a += dt * r * c * Max(InnerProduct(w, n), 0) * self._ds(marker)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:

        if self.fixed_velocity:
            L = CoefficientFunction(0) * dx
        else:
            # Calculate the hydrodynamic contribution to the linear terms
            L = super().construct_linear(V, gfu_0, dt, time_step)[0]

        if self.linearize == 'Oseen':
            w = self._get_wind(gfu_0, time_step)

        # Define the special DG functions.
        n, _, alpha, _ = get_special_functions(self.mesh, self.nu)

        # Transport of mixture components.
        for comp in self.extra_components:
            # Test function
            r = V[self.model_components[comp]]

            # NOTE: Reaction terms can go in either the bilinear form or the linear form depending on if the reaction
            #       coefficients are functions of the trial functions. Basic type-checking can't be used since the
            #       reaction coefficients will always be coefficientfunctions or gridfunctions. Instead, check if the
            #       reaction coefficients are gridfunctions (go in linear form) and if they are coefficientfunctions
            #       expand out the coefficientfunction tree and check for the presence of "trial-function" (goes in
            #       bilinear form).

            # Surface reaction
            for marker in self.BC.get('surface_rxn', {}).get(comp, {}):
                val = self.BC['surface_rxn'][comp][marker][time_step]
                if not isinstance(val, ProxyFunction) and (not isinstance(val, CoefficientFunction) or ('trial-function' not in val.__str__())):
                    if self.DG:
                        L += -dt * val * r * self._ds(marker)
                    else:
                        L += -dt * val * r.Trace() * self._ds(marker)

            # Bulk reaction
            # TODO: Temp solution to test, implement better once parser has been updated
            val = self.f[comp][time_step]
            if not isinstance(val, ProxyFunction) and (not isinstance(val, CoefficientFunction) or ('trial-function' not in val.__str__())):
                if self.BC.get('surface_rxn', {}).get(comp, None) is not None:
                    marker = list(self.BC['surface_rxn'][comp].keys())[0]
                    L += dt * val * r * self._ds(marker)
                else:
                    L += dt * val * r * dx

            # 1/2 of Total Flux BC ("Natural" BC), other half in bilinear term
            for marker in self.BC.get('total_flux', {}).get(comp, {}):
                h = self.BC['total_flux'][comp][marker][time_step]
                if self.DG:
                    L -= dt * r * h * self._ds(marker)
                else:
                    L -= dt * r.Trace() * h * self._ds(marker)

            # 1/2 of the Neumann BC, other half in the bilinear form
            for marker in self.BC.get('neumann', {}).get(comp, {}):
                if self.Ds[comp][time_step] == 0:
                    raise ValueError("Trying to apply a neuman boundary condition for "
                                     "purely advective flow (diffusion coefficient is 0).")
                h = self.BC['neumann'][comp][marker][time_step]
                # NOTE: Negative sign applied since moved to other side of equal sign
                if self.DG:
                    L -= dt * r * h * self._ds(marker)
                else:
                    L -= dt * r.Trace() * h * self._ds(marker)

            if self.DG:
                for marker in self.BC.get('dirichlet', {}).get(comp, {}):
                    g = self.BC['dirichlet'][comp][marker][time_step]

                    # 1/2 of diffusion on Dirichlet BC (1st line)
                    # AND
                    # 1/2 of penalty on Dirichlet BC (2nd line)
                    if self.Ds[comp][time_step] > 0:
                        L += dt * self.Ds[comp][time_step] * g * (
                                alpha * r
                                - InnerProduct(Grad(r), n)
                        ) * self._ds(marker)

                    # 1/2 of advection on Dirichlet BC
                    if self.linearize == 'Oseen':
                        L -= dt * r * g * Min(InnerProduct(w, n), 0) * self._ds(marker)

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:
        if self.fixed_velocity:
            L = CoefficientFunction(0) * dx
            # Velocity linearization term
            gfu_u = self.W[self.model_components_ic['u']]
        else:
            # Calculate the hydrodynamic contribution to the linear terms
            L = super().construct_imex_explicit(V, gfu_0, dt, time_step)[0]
            # Velocity linearization term
            gfu_u = gfu_0[self.model_components['u']]

        # Define the special DG functions.
        n, _, alpha, _ = get_special_functions(self.mesh, self.nu)

        for comp in self.extra_components:
            # Previous concentration result
            c_prev = gfu_0[self.model_components[comp]]

            # Test function
            r = V[self.model_components[comp]]

            # Convection term
            # Multiplied by a negative since brought to linear side
            L += dt * c_prev * InnerProduct(gfu_u, Grad(r)) * dx

            # 1/2 of Neuman BC
            # TODO: Is this implemented correctly?
            for marker in self.BC.get('neumann', {}).get(comp, {}):
                if self.Ds[comp][time_step] == 0:
                    raise ValueError("Trying to apply a neuman boundary condition for "
                                     "purely advective flow (diffusion coefficient is 0).")
                h = self.BC['neumann'][comp][marker][time_step]
                if self.DG:
                    L -= dt * r * h * self._ds(marker)
                else:
                    L -= dt * r.Trace() * h * self._ds(marker)

            # Check that a total flux BC has not been incorrectly specified
            # TODO: Is this implemented correctly?
            if self.BC.get('total_flux', {}).get(comp, None) is not None:
                raise ValueError('Cannot specify total flux for IMEX schemes since convection term is fully explicit.')

            if self.DG:
                for marker in self.BC.get('dirichlet', {}).get(comp, {}):
                    g = self.BC['dirichlet'][comp][marker][time_step]

                    # 1/2 of diffusion on Dirichlet BC (1st line)
                    # AND
                    # 1/2 of penalty on Dirichlet BC (2nd line)
                    if self.Ds[comp][time_step] > 0:
                        L += dt * self.Ds[comp][time_step] * g * (
                                alpha * r
                                - InnerProduct(Grad(r), n)
                        ) * self._ds(marker)

                    # 1/2 of advection on Dirichlet BC
                    L -= dt * r * g * Min(InnerProduct(gfu_u, n), 0) * self._ds(marker)

        return [L]

    def solve_single_step(self, a_lst: List[BilinearForm], L_lst: List[LinearForm],
                          precond_lst: List[Preconditioner], gfu: GridFunction, time_step: int = 0) -> None:
        if self.fixed_velocity:
            self.linear_solve(a_lst[0], L_lst[0], precond_lst[0], gfu)
        else:
            super().solve_single_step(a_lst, L_lst, precond_lst, gfu, time_step)
