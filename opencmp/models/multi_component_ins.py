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

from ..helpers.ngsolve_ import get_special_functions
from . import INS
from ..config_functions import ConfigParser
from typing import Dict, List, Optional, Union
from ngsolve import BilinearForm, FESpace, Grad, InnerProduct, LinearForm, GridFunction, dx, \
    Parameter, CoefficientFunction
from ngsolve.comp import ProxyFunction

import ngsolve

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

        # Number of default variables
        num_existing = len(self.model_components)

        for i in range(len(self.extra_components)):
            component = self.extra_components[i]

            self.model_components[component]                = i + num_existing
            self.model_local_error_components[component]    = new_components_in_error_calc[component]
            self.time_derivative_components[0][component]   = new_components_in_time_deriv[component]

    def _construct_fes(self) -> FESpace:
        # Create the FE spaces for velocity and pressure
        fes_total = self._contruct_fes_helper()

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
        if self.DIM or self.DG:
            raise NotImplementedError('DIM and DG are not yet implemented.')

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
        assert len(self.f) == len(self.extra_components) + 1  # Extra +1 since velocity also has a source term.

    def _get_wind(self, U, time_step):
        if self.fixed_velocity:
            return self.W[0]
        else:
            return super()._get_wind(U, time_step)

    def construct_bilinear_time_coefficient(self, U: List[ProxyFunction], V: List[ProxyFunction], dt: Parameter,
                                            time_step: int) -> List[BilinearForm]:

        # Calculate the hydrodynamic contribution to the bilinear terms
        a = super().construct_bilinear_time_coefficient(U, V, dt, time_step)[0]

        w = self._get_wind(U, time_step)

        # TODO: Not sure that this has actually been split up correctly.
        #       Convection term is currently the only term that must always be solved implicitly.
        # Transport of mixture components.
        if self.linearize == 'Oseen':
            for comp in self.extra_components:
                # Trial and test functions
                c = U[self.model_components[comp]]
                r = V[self.model_components[comp]]

                # Additional convection term.
                a += -dt * c * InnerProduct(w, Grad(r)) * dx

        return [a]

    def construct_bilinear_time_ODE(self, U: Union[List[ProxyFunction], List[GridFunction]], V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0), time_step: int = 0) -> List[BilinearForm]:

        # Calculate the hydrodynamic contribution to the bilinear terms
        a = super().construct_bilinear_time_ODE(U, V, dt, time_step)[0]

        w = self._get_wind(U, time_step)

        # Get unit normal
        n = get_special_functions(self.mesh, self.nu)[0]

        # Transport of the mixture components.
        for comp in self.extra_components:
            # Trial and test functions
            c = U[self.model_components[comp]]
            r = V[self.model_components[comp]]

            # Domain integrals.
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
                    a += -dt * val * r * self._ds(marker)

            # Bulk reaction
            # TODO: Temp solution to test, implement better once parser has been updated
            val = self.f[comp][time_step]
            if isinstance(val, ProxyFunction) or (isinstance(val, CoefficientFunction) and ('trial-function' in val.__str__())):
                # NOTE: The negative sign is here since the source term was brought over to bilinear side, it usually
                #       does not have one since it is on the linear term side.
                if self.BC.get('surface_rxn', {}).get(comp, None) is not None:
                    marker = list(self.BC['surface_rxn'][comp].keys())[0]
                    a += -dt * val * r * self._ds(marker)
                else:
                    a += -dt * val * r * dx

            if self.linearize == 'Oseen':
                # Neumann BC for C
                for marker in self.BC.get('neumann', {}).get(comp, {}):
                    h = self.BC['neumann'][comp][marker][time_step]
                    a += -dt * r.Trace() * (h - c * InnerProduct(w, n)) * self._ds(marker)

        return [a]

    def construct_linear(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                         dt: Parameter, time_step: int) -> List[LinearForm]:

        # Calculate the hydrodynamic contribution to the linear terms
        L = super().construct_linear(V, gfu_0, dt, time_step)[0]

        # Transport of mixture components.
        for comp in self.extra_components:
            # Test function
            r = V[self.model_components[comp]]

            # Total Flux BC
            for marker in self.BC.get('total_flux', {}).get(comp, {}):
                h = self.BC['total_flux'][comp][marker][time_step]
                L += dt * r.Trace() * h * self._ds(marker)

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
                    L += -dt * val * r * self._ds(marker)

            # Bulk reaction
            # TODO: Temp solution to test, implement better once parser has been updated
            val = self.f[comp][time_step]
            if not isinstance(val, ProxyFunction) and (not isinstance(val, CoefficientFunction) or ('trial-function' not in val.__str__())):
                if self.BC.get('surface_rxn', {}).get(comp, None) is not None:
                    marker = list(self.BC['surface_rxn'][comp].keys())[0]
                    L += dt * val * r * self._ds(marker)
                else:
                    L += dt * val * r * dx

        return [L]

    def construct_imex_explicit(self, V: List[ProxyFunction], gfu_0: Optional[List[GridFunction]],
                                dt: Parameter, time_step: int) -> List[LinearForm]:

        # Calculate the hydrodynamic contribution to the linear terms
        L = super().construct_imex_explicit(V, gfu_0, dt, time_step)[0]

        # Velocity linearization term
        gfu_u = gfu_0[self.model_components['u']]

        for comp in self.extra_components:
            # Previous concentration result
            c_prev = gfu_0[self.model_components[comp]]

            # Test function
            r = V[self.model_components[comp]]

            # Convection term
            L += -dt * c_prev * InnerProduct(gfu_u, Grad(r)) * dx

            # Neumann BC for C
            # TODO: Is this implemented correctly?
            for marker in self.BC.get('neumann', {}).get(comp, {}):
                h = self.BC['neumann'][comp][marker][time_step]
                L += -dt * r.Trace() * h * self._ds(marker)

            # Check that a total flux BC has not been incorrectly specified
            # TODO: Is this implemented correctly?
            if self.BC.get('total_flux', {}).get(comp, None) is not None:
                raise ValueError('Cannot specify total flux for IMEX schemes since convection term is fully explicit.')

        return [L]
