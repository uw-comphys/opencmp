import ngsolve as ngs
from helpers.ngsolve_ import get_special_functions, construct_p_mat
from helpers.dg import avg, jump, grad_avg
from models import Model
from config_functions import ConfigParser
from typing import Tuple, List, Optional, Union
from ngsolve.comp import FESpace, ProxyFunction, GridFunction
from helpers.error import norm, mean_to_zero


class INS(Model):
    """
    A single phase incompressible Navier-Stokes model.
    """

    def __init__(self, config: ConfigParser, t_param: ngs.Parameter) -> None:
        # Set dictionary to map variable name to index inside fes/gridfunctions
        # Also used for labeling variables in vtk output.
        # NOTE: This MUST be set before calling super(), since it's needed in superclass' __init__
        self.model_components = {'u': 0, 'p': 1}

        self.model_local_error_components = {'u': True, 'p': True}

        # Pre-define which BCs are accepted for this model, all others are thrown out.
        self.BC_init = {'dirichlet': {},
                        'neumann': {},
                        'stress': {},
                        'no_backflow': {},
                        'parallel': {}}

        super().__init__(config, t_param)

        # Load the solver parameters.
        self.linearize = self.config.get_item(['SOLVER', 'linearization_method'], str)
        if self.linearize not in ['Oseen', 'IMEX']:
            raise TypeError('Don\'t recognize the linearization method.')

        if self.linearize == 'Oseen':
            self.nonlinear_tolerance = self.config.get_item(['SOLVER', 'nonlinear_tolerance'], float)
            self.nonlinear_max_iters = self.config.get_item(['SOLVER', 'nonlinear_max_iterations'], int)
            self.W = self._construct_linearization_terms()

    @staticmethod
    def allows_explicit_schemes() -> bool:
        # INS cannot work with explicit schemes
        return False

    def _set_model_parameters(self) -> None:
        self.kv = self.model_functions.model_parameters_dict['kinematic_viscosity']
        self.f = self.model_functions.model_functions_dict['source']['u']

    def _construct_fes(self) -> FESpace:
        if self.element[0] == 'RT':
            # Raviart-Thomas elements are a type of HDiv finite element.
            fes_u = ngs.HDiv(self.mesh, order=self.interp_ord, dirichlet=self.dirichlet_names.get('u', ''),
                             dgjumps=self.DG, RT=True)
        else:
            fes_u = getattr(ngs, self.element[0])(self.mesh, order=self.interp_ord,
                                                  dirichlet=self.dirichlet_names.get('u', ''), dgjumps=self.DG)

        fes_p = getattr(ngs, self.element[1])(self.mesh, order=self.interp_ord - 1, dgjumps=self.DG)

        return ngs.FESpace([fes_u, fes_p], dgjumps=self.DG)

    def _construct_linearization_terms(self) -> Optional[List[ngs.GridFunction]]:
        tmp = ngs.GridFunction(self.fes.components[0])
        tmp.vec.data = self.IC.components[0].vec

        return [tmp]

    def construct_bilinear(self, U: List[ProxyFunction], V: List[ProxyFunction],
                           explicit_bilinear: bool = False) -> ngs.BilinearForm:
        u, p = U
        v, q = V

        if self.linearize == 'IMEX':
            # Use an IMEX operator-splitting scheme to linearize the convection term.
            if self.DIM:
                # Use the diffuse interface method.
                a = self._bilinear_IMEX_DIM(u, p, v, q, explicit_bilinear)
            else:
                # Solve on a standard conformal mesh.
                a = self._bilinear_IMEX_no_DIM(u, p, v, q, explicit_bilinear)

        elif self.linearize == 'Oseen':
            # Solve the Oseen equations, a linearized form of INS.
            if self.DIM:
                # Use the diffuse interface method.
                a = self._bilinear_Oseen_DIM(u, p, v, q, explicit_bilinear)
            else:
                # Solve on a standard conformal mesh.
                a = self._bilinear_Oseen_no_DIM(u, p, v, q, explicit_bilinear)

        return a

    def construct_linear(self, V: List[ProxyFunction], gfu_0: List[Union[GridFunction, None]] = None) \
            -> ngs.BilinearForm:
        v, q = V

        if self.linearize == 'IMEX':
            # Use an IMEX operator-splitting scheme to linearize the convection term.
            gfu_u, gfu_p = gfu_0

            if self.DIM:
                # Use the diffuse interface method.
                L = self._linear_IMEX_DIM(v, gfu_u)
            else:
                # Solve on a standard conformal mesh.
                L = self._linear_IMEX_no_DIM(v, gfu_u)

        elif self.linearize == 'Oseen':
            # Solve the Oseen equations, a linearized form of INS.
            w = self.W[0]

            if self.DIM:
                # Use the diffuse interface method.
                L = self._linear_Oseen_DIM(v, w)
            else:
                # Solve on a standard conformal mesh.
                L = self._linear_Oseen_no_DIM(v, w)

        return L

    def get_trial_and_test_functions(self) -> Tuple[List[ProxyFunction], List[ProxyFunction]]:
        # Define the test and trial functions.
        u, p = self.fes.TrialFunction()
        v, q = self.fes.TestFunction()

        return [u, p], [v, q]

    def single_iteration(self, a: ngs.BilinearForm, L: ngs.LinearForm, precond: ngs.Preconditioner,
                         gfu: ngs.GridFunction) -> None:

        if self.linearize == 'Oseen':
            self.construct_and_run_solver(a, L, precond, gfu)
            component = self.model_components['u']
            err = norm("l2_norm", self.W[0], gfu.components[0], self.mesh, self.fes.components[component], average=False)
            numit = 1

            if self.verbose > 0:
                print(numit, err)

            while (err > self.nonlinear_tolerance) and (numit < self.nonlinear_max_iters):
                self.W[0].vec.data = gfu.components[0].vec
                if not 'p' in self.BC['dirichlet'].keys():
                    # Bias mean of pressure to zero.
                    p_biased = mean_to_zero(gfu.components[1], self.fes.components[1], self.mesh)
                    gfu.components[1].vec.data = p_biased.vec
                self.apply_dirichlet_bcs_to(gfu)

                a.Assemble()
                L.Assemble()
                precond.Update()

                self.construct_and_run_solver(a, L, precond, gfu)

                err = norm("l2_norm", self.W[0], gfu.components[0], self.mesh, self.fes.components[component], average=False)
                numit += 1

                if self.verbose > 0:
                    print(numit, err)

            self.W[0].vec.data = gfu.components[0].vec
            if not 'p' in self.BC['dirichlet'].keys():
                # Bias mean of pressure to zero.
                p_biased = mean_to_zero(gfu.components[1], self.fes.components[1], self.mesh)
                gfu.components[1].vec.data = p_biased.vec

        elif self.linearize == 'IMEX':
            self.construct_and_run_solver(a, L, precond, gfu)
            if not 'p' in self.BC['dirichlet'].keys():
                # Bias mean of pressure to zero.
                p_biased = mean_to_zero(gfu.components[1], self.fes.components[1], self.mesh)
                gfu.components[1].vec.data = p_biased.vec

########################################################################################################################
# BILINEAR AND LINEAR FORM HELPER FUNCTIONS
########################################################################################################################

    def _bilinear_Oseen_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction,
                            explicit_bilinear) -> ngs.BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used with Oseen linearization. Handles both CG and DG.
        """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        p_I = construct_p_mat(p, self.mesh.dim)

        w = self.W[0]

        a = (
            self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            - ngs.InnerProduct(ngs.OuterProduct(u, w), ngs.Grad(v))  # Convection term
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u and grad(p) to zero where phi is zero.
        a += (
            alpha * u * v # Removing the alpha penalty following discussion with James.
            - p * (ngs.div(v))
            ) * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(n, u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(n, v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += (
                    jump_v * (w * n * avg_u + 0.5 * ngs.Norm(w * n) * jump_u)  # Convection
                    - self.kv * ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                    - self.kv * ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                    + self.kv * alpha * ngs.InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I
                                                                          # from ∇u^
                    ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += (
                    v * (0.5 * w * n * u + 0.5 * ngs.Norm(w * n) * u)  # 1/2 of uw^ (convection term)
                    - self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on
                    + self.kv * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += (
                v * (0.5 * w * -self.DIM_solver.grad_phi_gfu * u + 0.5 * ngs.Norm(w * -self.DIM_solver.grad_phi_gfu) * u)
                + self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, self.DIM_solver.grad_phi_gfu))
                + self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, self.DIM_solver.grad_phi_gfu))
                + self.kv * alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Dirichlet for p
        # Add the convection and stress terms on the bilinear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            if self.DG:
                a += v * ((ngs.OuterProduct(u, w) - self.kv * ngs.Grad(u)) * n) * self._ds(marker)
            else:
                a += v.Trace() * ((ngs.OuterProduct(u.Trace(), w) - self.kv * ngs.Grad(u.Trace())) * n) * self._ds(
                    marker)

        # Conformal Neumann for u
        # Add convection and pressure terms to bilinear form
        for marker in self.BC.get('neumann', {}).get('u', {}):
            if self.DG:
                a += v * ((ngs.OuterProduct(u, w) + p_I) * n) * self._ds(marker)
            else:
                a += v.Trace() * ((ngs.OuterProduct(u.Trace(), w) + p_I) * n) * self._ds(marker)

        # Conformal no backflow
        for marker in self.BC.get('no_backflow', {}).get('no_backflow', {}):
            if self.DG:
                a += v * (ngs.IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
            else:
                a += v.Trace() * (ngs.IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Conformal parallel flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += v * (u - n * ngs.InnerProduct(u, n)) * self._ds(marker)
            else:
                a += v.Trace() * (u.Trace() - n * ngs.InnerProduct(u.Trace(), n)) * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return a

    def _bilinear_IMEX_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction,
                            explicit_bilinear) -> ngs.BilinearForm:
        """
        Bilinear form when the diffuse interface method is being used with IMEX linearization. Handles both CG and DG.
        """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        p_I = construct_p_mat(p, self.mesh.dim)

        a = (
            self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Force u and grad(p) to zero where phi is zero.
        a += (
            alpha * u * v # Removing the alpha penalty following discussion with James.
            - p * (ngs.div(v))
            ) * (1.0 - self.DIM_solver.phi_gfu) * ngs.dx

        if self.DG:
            jump_u = jump(n, u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(n, v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += (
                    - self.kv * ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                    - self.kv * ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                    + self.kv * alpha * ngs.InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I
                                                                          # from ∇u^
                     ) * self.DIM_solver.phi_gfu * ngs.dx(skeleton=True)

            if self.dirichlet_names.get('u', None) is not None:
                # Penalty terms for conformal Dirichlet BCs
                a += (
                    - self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on
                    + self.kv * alpha * u * v  # 1/2 of penalty term for u=g on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            a += (
                self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, self.DIM_solver.grad_phi_gfu))
                + self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, self.DIM_solver.grad_phi_gfu))
                + self.kv * alpha * u * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Dirichlet for p
        # Add the stress term to the bilinear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            if self.DG:
                a += v * ((-self.kv * ngs.Grad(u)) * n) * self._ds(marker)
            else:
                a += v.Trace() * ((-self.kv * ngs.Grad(u.Trace())) * n) * self._ds(marker)

        # Conformal Neumann for u
        # Add the pressure term to bilinear form
        for marker in self.BC.get('neumann', {}).get('u', {}):
            if self.DG:
                a += v * (p_I * n) * self._ds(marker)
            else:
                a += v.Trace() * (p_I * n) * self._ds(marker)

        # Conformal parallel flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += v * (u - n * ngs.InnerProduct(u, n)) * self._ds(marker)
            else:
                a += v.Trace() * (u.Trace() - n * ngs.InnerProduct(u.Trace(), n)) * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return a

    def _bilinear_Oseen_no_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction,
                            explicit_bilinear) -> ngs.BilinearForm:
        """
        Bilinear form when Oseen linearization is being used and the diffuse interface method is not being used.
        Handles both CG and DG.
        """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        p_I = construct_p_mat(p, self.mesh.dim)

        w = self.W[0]

        a = (
            self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            - ngs.InnerProduct(ngs.OuterProduct(u, w), ngs.Grad(v))  # Convection term
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            ) * ngs.dx

        if self.DG:
            avg_u = avg(u)
            jump_u = jump(n, u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(n, v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += (
                    jump_v * (w * n * avg_u + 0.5 * ngs.Norm(w * n) * jump_u)  # Convection
                    - self.kv * ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                    - self.kv * ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                    + self.kv * alpha * ngs.InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I
                                                                          # from ∇u^
                    ) * ngs.dx(skeleton=True)

            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += (
                    v * (0.5 * w * n * u + 0.5 * ngs.Norm(w * n) * u)  # 1/2 of uw^ (convection term)
                    - self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on
                    + self.kv * alpha * u * v                                          # 1/2 of penalty term for u=g
                                                                                       # on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Dirichlet for p
        # Add the convection and stress terms on the bilinear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            if self.DG:
                a += v * ((ngs.OuterProduct(u, w) - self.kv * ngs.Grad(u)) * n) * self._ds(marker)
            else:
                a += v.Trace() * ((ngs.OuterProduct(u.Trace(), w) - self.kv * ngs.Grad(u.Trace())) * n) * self._ds(
                    marker)

        # Neumann for u
        # Add convection and pressure terms to bilinear form
        for marker in self.BC.get('neumann', {}).get('u', {}):
            if self.DG:
                a += v * ((ngs.OuterProduct(u, w) + p_I) * n) * self._ds(marker)
            else:
                a += v.Trace() * ((ngs.OuterProduct(u.Trace(), w) + p_I) * n) * self._ds(marker)

        # No backflow
        for marker in self.BC.get('no_backflow', {}).get('no_backflow', {}):
            if self.DG:
                a += v * (ngs.IfPos(w * n, w * n, 0.0) * u) * self._ds(marker)
            else:
                a += v.Trace() * (ngs.IfPos(w * n, w * n, 0.0) * u.Trace()) * self._ds(marker)

        # Parallel Flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += v * (u - n * ngs.InnerProduct(u, n)) * self._ds(marker)
            else:
                a += v.Trace() * (u.Trace() - n * ngs.InnerProduct(u.Trace(), n)) * self._ds(marker)

        return a

    def _bilinear_IMEX_no_DIM(self, u: ProxyFunction, p: ProxyFunction, v: ProxyFunction, q: ProxyFunction,
                            explicit_bilinear) -> ngs.BilinearForm:
        """
        Bilinear form when IMEX linearization is being used and the diffuse interface method is not being used.
        Handles both CG and DG.
        """

        # Define the special DG functions.
        n, _, alpha = get_special_functions(self.mesh, self.nu)

        p_I = construct_p_mat(p, self.mesh.dim)

        a = (
            self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v))  # Stress, Newtonian
            - ngs.div(u) * q  # Conservation of mass
            - ngs.div(v) * p  # Pressure
            ) * ngs.dx

        if self.DG:
            jump_u = jump(n, u)
            avg_grad_u = grad_avg(u)

            jump_v = jump(n, v)
            avg_grad_v = grad_avg(v)

            if not explicit_bilinear:
                # Penalty for discontinuities
                a += (
                    - self.kv * ngs.InnerProduct(avg_grad_u, ngs.OuterProduct(jump_v, n))  # Stress
                    - self.kv * ngs.InnerProduct(avg_grad_v, ngs.OuterProduct(jump_u, n))  # U
                    + self.kv * alpha * ngs.InnerProduct(jump_u, jump_v)  # Penalty term for u+=u- on 𝚪_I
                                                                          # from ∇u^
                    ) * ngs.dx(skeleton=True)

            # Penalty for dirichlet BCs
            if self.dirichlet_names.get('u', None) is not None:
                a += (
                    - self.kv * ngs.InnerProduct(ngs.Grad(u), ngs.OuterProduct(v, n))  # ∇u^ = ∇u
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(u, n))  # 1/2 of penalty for u=g on
                    + self.kv * alpha * u * v                                          # 1/2 of penalty term for u=g
                                                                                       # on 𝚪_D from ∇u^
                    ) * self._ds(self.dirichlet_names['u'])

        # Dirichlet for p
        # Add the stress term to the bilinear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            if self.DG:
                a += v * ((-self.kv * ngs.Grad(u)) * n) * self._ds(marker)
            else:
                a += v.Trace() * ((-self.kv * ngs.Grad(u.Trace())) * n) * self._ds(marker)

        # Neumann for u
        # Add the pressure term to bilinear form
        for marker in self.BC.get('neumann', {}).get('u', {}):
            if self.DG:
                a += v * (p_I * n) * self._ds(marker)
            else:
                a += v.Trace() * (p_I * n) * self._ds(marker)

        # Parallel Flow BC
        for marker in self.BC.get('parallel', {}).get('parallel', {}):
            if self.DG:
                a += v * (u - n * ngs.InnerProduct(u, n)) * self._ds(marker)
            else:
                a += v.Trace() * (u.Trace() - n * ngs.InnerProduct(u.Trace(), n)) * self._ds(marker)

        return a

    def _linear_Oseen_DIM(self, v: ProxyFunction, w: GridFunction) -> ngs.LinearForm:
        """
        Linear form when the diffuse interface method is being used with Oseen linearization. Handles both CG and DG.
        """

        L = v * self.f * self.DIM_solver.phi_gfu * ngs.dx

        # Define the special DG functions.
        n, h, alpha = get_special_functions(self.mesh, self.nu)

        if self.DG:
            # Conformal Dirichlet BCs for u.
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += (
                    v * (-0.5 * w * n * g + 0.5 * ngs.Norm(w * n) * g)  # 1/2 of uw^ (convection)
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    + self.kv * alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    ) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            # Penalty terms for DIM Dirichlet BCs
            g = self.DIM_BC['dirichlet']['u'][marker]
            L += (
                v * (0.5 * w * self.DIM_solver.grad_phi_gfu * g + 0.5 * ngs.Norm(w * -self.DIM_solver.grad_phi_gfu) * g)
                + self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, self.DIM_solver.grad_phi_gfu))
                + self.kv * alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
            ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Neumann BC for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker]
            if self.DG:
                L += v * self.kv * h * self._ds(marker)
            else:
                L += v.Trace() * self.kv * h * self._ds(marker)

        # Conformal Dirichlet BC for p
        # Very weakly impose pressure dirichlet values by adding them to the linear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            p_D = self.BC['dirichlet']['p'][marker]
            p_I = construct_p_mat(p_D, self.mesh.dim)

            if self.DG:
                L += v * (-p_I * n) * ngs.ds(skeleton=True, definedon=marker)
            else:
                L += v.Trace() * (-p_I * n) * ngs.ds(definedon=marker)

        # Conformal stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker]
            if self.DG:
                L += v * h * self._ds(marker)
            else:
                L += v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return L

    def _linear_IMEX_DIM(self, v: ProxyFunction, gfu_u: GridFunction) -> ngs.LinearForm:
        """
        Linear form when the diffuse interface method is being used with IMEX linearization. Handles both CG and DG.
        """

        L = (
            v * self.f - ngs.InnerProduct(ngs.Grad(gfu_u) * gfu_u, v)
            ) * self.DIM_solver.phi_gfu * ngs.dx

        # Define the special DG functions.
        n, h, alpha = get_special_functions(self.mesh, self.nu)

        if self.DG:
            # Conformal Dirichlet BCs for u.
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += (
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    + self.kv * alpha * g * v  # 1/2 of penalty for u=g from ∇u^ on 𝚪_D
                    ) * self._ds(marker)

        # Penalty term for DIM Dirichlet BCs. This is the Nitsche method.
        for marker in self.DIM_BC.get('dirichlet', {}).get('u', {}):
            # Penalty terms for DIM Dirichlet BCs
            g = self.DIM_BC['dirichlet']['u'][marker]
            L += (
                self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, self.DIM_solver.grad_phi_gfu))
                + self.kv * alpha * g * v * self.DIM_solver.mag_grad_phi_gfu
                ) * self.DIM_solver.mask_gfu_dict[marker] * ngs.dx

        # Conformal Neumann BC for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker]
            if self.DG:
                L += v * self.kv * h * self._ds(marker)
            else:
                L += v.Trace() * self.kv * h * self._ds(marker)

        # Conformal Dirichlet BC for p
        # Very weakly impose pressure dirichlet values by adding them to the linear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            p_D = self.BC['dirichlet']['p'][marker]
            p_I = construct_p_mat(p_D, self.mesh.dim)

            if self.DG:
                L += v * (-p_I * n) * ngs.ds(skeleton=True, definedon=marker)
            else:
                L += v.Trace() * (-p_I * n) * ngs.ds(definedon=marker)

        # Conformal stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker]
            if self.DG:
                L += v * h * self._ds(marker)
            else:
                L += v.Trace() * h * self._ds(marker)

        # TODO: Add non-Dirichlet DIM BCs.

        return L

    def _linear_Oseen_no_DIM(self, v: ProxyFunction, w: GridFunction) -> ngs.LinearForm:
        """
        Linear form when Oseen linearization is being used and the diffuse interface method is not being used.
        Handles both CG and DG.
        """

        L = v * self.f * ngs.dx

        # Define the special DG functions.
        n, h, alpha = get_special_functions(self.mesh, self.nu)

        # Dirichlet BC for u
        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += (
                    v * (-0.5 * w * n * g + 0.5 * ngs.Norm(w * n) * g)  # 1/2 of uw^ (convection)
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    + self.kv * alpha * g * v                                          # 1/2 of penalty for u=g
                                                                                       # from ∇u^ on 𝚪_D
                ) * self._ds(marker)

        # Neumann BC for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker]
            if self.DG:
                L += v * self.kv * h * self._ds(marker)
            else:
                L += v.Trace() * self.kv * h * self._ds(marker)

        # Dirichlet BC for p
        # Very weakly impose pressure dirichlet values by adding them to the linear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            p_D = self.BC['dirichlet']['p'][marker]
            p_I = construct_p_mat(p_D, self.mesh.dim)

            if self.DG:
                L += v * (-p_I * n) * ngs.ds(skeleton=True, definedon=marker)
            else:
                L += v.Trace() * (-p_I * n) * ngs.ds(definedon=marker)

        # Stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker]
            if self.DG:
                L += v * h * self._ds(marker)
            else:
                L += v.Trace() * h * self._ds(marker)

        return L

    def _linear_IMEX_no_DIM(self, v: ProxyFunction, gfu_u: GridFunction) -> ngs.LinearForm:
        """
        Linear form when IMEX linearization is being used and the diffuse interface method is not being used.
        Handles both CG and DG.
        """

        L = (
            v * self.f - ngs.InnerProduct(ngs.Grad(gfu_u) * gfu_u, v)
            ) * ngs.dx

        # Define the special DG functions.
        n, h, alpha = get_special_functions(self.mesh, self.nu)

        # Dirichlet BC for u
        if self.DG:
            for marker in self.BC.get('dirichlet', {}).get('u', {}):
                g = self.BC['dirichlet']['u'][marker]
                L += (
                    - self.kv * ngs.InnerProduct(ngs.Grad(v), ngs.OuterProduct(g, n))  # 1/2 of penalty for u=g
                    + self.kv * alpha * g * v                                          # 1/2 of penalty for u=g
                                                                                       # from ∇u^ on 𝚪_D
                    ) * self._ds(marker)

        # Neumann BC for u
        for marker in self.BC.get('neumann', {}).get('u', {}):
            h = self.BC['neumann']['u'][marker]
            if self.DG:
                L += v * self.kv * h * self._ds(marker)
            else:
                L += v.Trace() * self.kv * h * self._ds(marker)

        # Dirichlet BC for p
        # Very weakly impose pressure dirichlet values by adding them to the linear form
        for marker in self.BC.get('dirichlet', {}).get('p', {}):
            p_D = self.BC['dirichlet']['p'][marker]
            p_I = construct_p_mat(p_D, self.mesh.dim)

            if self.DG:
                L += v * (-p_I * n) * ngs.ds(skeleton=True, definedon=marker)
            else:
                L += v.Trace() * (-p_I * n) * ngs.ds(definedon=marker)

        # Stress BC
        for marker in self.BC.get('stress', {}).get('stress', {}):
            h = self.BC['stress']['stress'][marker]
            if self.DG:
                L += v * h * self._ds(marker)
            else:
                L += v.Trace() * h * self._ds(marker)

        return L