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

"""Geometry-independent wall functions for the k-epsilon turbulence model."""

import logging
import numpy as np
import ngsolve as ngs


def wall_distance(mesh: ngs.comp.Mesh, wall_boundary: str = 'wall',
                  order: int = 2, relax: float = 0.1) -> ngs.GridFunction:
    """
    Distance to ``wall_boundary`` from a regularized Eikonal solve.

    Free of any turbulence modelling, so models that need only a wall distance
    (e.g. the two-fluid lift wall-deactivation taper) can use it without
    constructing a :class:`KEpsilonWallFunction`.

    Args:
        mesh: The mesh used for the simulation.
        wall_boundary: Boundary marker(s) to measure the distance from.
        order: Order of the H1 space the distance field lives in.
        relax: Regularization, as a multiple of the local mesh size.

    Returns:
        The wall-distance field.
    """

    eps = relax * ngs.specialcf.mesh_size
    fes = ngs.H1(mesh, order=order, dirichlet=wall_boundary)
    u, v = fes.TnT()
    y = ngs.GridFunction(fes)

    a = ngs.BilinearForm(fes)
    a += ngs.grad(u) * ngs.grad(v) * ngs.dx
    f = ngs.LinearForm(fes)
    f += 1.0 * v * ngs.dx
    a.Assemble()
    f.Assemble()
    y.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

    gu = ngs.grad(u)
    residual = ngs.BilinearForm(fes)
    residual += (
        ngs.sqrt(gu * gu + 1e-12) * v - v
        + eps * gu * ngs.grad(v)
    ) * ngs.dx
    ngs.solvers.Newton(residual, y, printing=False)
    return y


class KEpsilonWallFunction:
    """Wall-layer eddy viscosity and dissipation for high-Re k-epsilon.

    The wall layer is found from mesh topology: cells owning a facet on
    ``wall_boundary``. ``eval_nu_t`` puts those cells on the wall law and every
    other cell on bulk
    ``C_mu k^2/epsilon``.

    Friction velocity is either ``C_mu**0.25 * sqrt(k)`` or the square root of
    the resolved tangential wall traction per unit density. Both are stored as
    P0 data on wall-facet owners. ``update`` refreshes them once per Picard
    iteration.

    Wall distance comes from a regularized-Eikonal solve: the continuous field
    for coefficients integrated across a cell (``y_plus_field``,
    ``eval_nu_wall``), and its cell average for per-cell quantities
    (``y_plus_cell``, ``epsilon_wall_cell``). Turbulence production is not
    handled here -- ``KEpsilonINS`` owns it.
    """

    #: y+ below which the viscous sublayer is assumed (no wall eddy viscosity).
    YPLUS_VISCOUS = 11.25
    #: Recommended range for the equilibrium log-law wall treatment.
    YPLUS_RECOMMENDED_MIN = 30.0
    YPLUS_RECOMMENDED_MAX = 300.0

    def __init__(self, mesh: ngs.comp.Mesh, nu: float, C_mu: float, kappa: float,
                 E_log: float = 9.8, wall_boundary: str = "wall",
                 dist_order: int = 2, dist_relax: float = 0.1,
                 u_tau_method: int = 0) -> None:
        self.mesh = mesh
        self.nu = nu                      # laminar kinematic viscosity
        self.C_mu = C_mu
        self.kappa = kappa
        self.E_log = E_log                # log-law roughness constant E
        self.wall_boundary = wall_boundary
        if u_tau_method not in (0, 1):
            raise ValueError('u_tau_method must be 0 (k-based) or 1 (velocity-based).')
        self.u_tau_method = u_tau_method
        self._warned_yplus_low = False
        self._warned_yplus_high = False
        self.h = ngs.specialcf.mesh_size

        # Piecewise-constant space shared by the masks, the cell distance and u_tau.
        self._fes0 = ngs.L2(mesh, order=0)

        self._mark_wall_cells()

        self._dist_gf = self._compute_distance_field(dist_order, dist_relax)
        self._dist_cell = ngs.GridFunction(self._fes0)
        self._dist_cell.Set(self._dist_gf)
        # y+ and epsilon_wall_cell both divide by this; guard against a Newton
        # undershoot on a degenerate cell producing a negative distance.
        distance = self._dist_cell.vec.FV().NumPy()
        distance[:] = np.maximum(distance, 1e-12)

        # Zero until the first update(); eval_nu_t is then simply bulk everywhere.
        self.u_tau_cell = ngs.GridFunction(self._fes0)
        self.wall_nu_t_cell = ngs.GridFunction(self._fes0)
        self.wall_shear_cell = ngs.GridFunction(self._fes0)
        self._y_plus_cell_gf = ngs.GridFunction(self._fes0)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _mark_wall_cells(self) -> None:
        """Find the wall layer from mesh topology.

        Sets ``_wall_measure``, ``_marked`` / ``wall_facet_cell_mask`` and
        ``mask`` to the cells that own a physical wall facet.
        """
        # Assumes a simplicial mesh (one L2(0) DOF per cell). Quads/hexes parse
        # but the resulting layer is untested, so refuse rather than go silently wrong.
        element_types = {element.type for element in self.mesh.Elements(ngs.VOL)}
        if not element_types <= {ngs.ET.TRIG, ngs.ET.TET}:
            raise NotImplementedError(
                'KEpsilonWallFunction supports simplicial meshes only (TRIG in 2D, '
                f'TET in 3D); this mesh contains {sorted(t.name for t in element_types)}.')

        # An L2 basis function has no boundary trace, so plain ds() assembles to
        # zero; skeleton=True evaluates it on the facet instead.
        lf = ngs.LinearForm(self._fes0)
        lf += self._fes0.TestFunction() * ngs.ds(
            definedon=self.mesh.Boundaries(self.wall_boundary), skeleton=True)
        lf.Assemble()
        self._wall_measure = np.array(lf.vec).ravel()

        # Wall-facet owners: the only cells where u_tau can be evaluated directly,
        # and where the algebraic epsilon condition is anchored.
        self._marked = self._wall_measure > 0.0
        if not self._marked.any():
            raise ValueError(
                f"KEpsilonWallFunction: boundary marker '{self.wall_boundary}' has "
                f"no boundary elements. Available boundaries: "
                f"{self.mesh.GetBoundaries()}.")
        self.wall_facet_cell_mask = ngs.GridFunction(self._fes0)
        self.wall_facet_cell_mask.vec.FV().NumPy()[:] = self._marked.astype(float)

        # A wall function is a first-cell treatment. Cells that do not own a
        # physical wall facet use the bulk closure, even when facet-adjacent to
        # a wall owner.
        self.mask = ngs.GridFunction(self._fes0)
        self.mask.vec.FV().NumPy()[:] = self._marked.astype(float)
        self._wall_layer_sources = {}
        self._element_dofs = {
            element.nr: tuple(element.dofs)
            for element in self._fes0.Elements(ngs.VOL)
        }
        self._wall_facets = self._find_physical_wall_facets()

    def _find_physical_wall_facets(self):
        """Map each wall facet to its owner, direction and geometric distance.

        At a corner, the two facets remain separate. Their friction velocities
        are combined only after projecting velocity with each facet's own normal.
        """
        owner_by_vertices = {}
        volume_elements = list(self.mesh.Elements(ngs.VOL))
        element_by_number = {element.nr: element for element in volume_elements}
        for element in volume_elements:
            for facet_id in element.facets:
                numbers = tuple(sorted(
                    vertex.nr for vertex in self.mesh[facet_id].vertices))
                owner_by_vertices.setdefault(numbers, []).append(element.nr)

        result = []
        for facet in self.mesh.Elements(ngs.BND):
            if facet.mat != self.wall_boundary:
                continue
            numbers = tuple(sorted(vertex.nr for vertex in facet.vertices))
            owners = owner_by_vertices.get(numbers, ())
            if len(owners) != 1:
                raise ValueError(
                    f'Wall facet {numbers} has {len(owners)} volume owners; '
                    'expected exactly one.')
            points = [np.asarray(self.mesh.vertices[number].point[:self.mesh.dim],
                                 dtype=float) for number in numbers]
            owner = element_by_number[owners[0]]
            centroid = np.mean([
                np.asarray(self.mesh.vertices[vertex.nr].point[:self.mesh.dim],
                           dtype=float)
                for vertex in owner.vertices
            ], axis=0)
            if self.mesh.dim == 2:
                direction = points[1] - points[0]
                direction /= np.linalg.norm(direction)  # unit tangent
                normal = np.asarray((-direction[1], direction[0]))
                distance = abs(float(np.dot(centroid - points[0], normal)))
            else:
                direction = np.cross(points[1] - points[0], points[2] - points[0])
                direction /= np.linalg.norm(direction)  # unit normal
                distance = abs(float(np.dot(centroid - points[0], direction)))
            result.append((owners[0], direction, max(distance, 1e-12)))
        return result

    def _compute_distance_field(self, order: int, relax: float) -> ngs.GridFunction:
        """Distance to ``wall_boundary`` from a regularized Eikonal solve."""
        return wall_distance(self.mesh, self.wall_boundary, order, relax)

    # ------------------------------------------------------------------
    # Per-iteration update
    # ------------------------------------------------------------------

    def _resolved_wall_shear(self, U) -> np.ndarray:
        """Facet-average molecular tangential traction for each wall cell.

        The tangential projection removes pressure and the isotropic part of
        the deviatoric stress. Taking the magnitude before facet integration
        prevents cancellation at corners while retaining a local value instead
        of imposing a streamwise/global average.
        """
        n = ngs.specialcf.normal(self.mesh.dim)
        grad_u = ngs.grad(U)
        traction = self.nu * (grad_u + grad_u.trans) * n
        tangential = traction - (traction * n) * n

        test = self._fes0.TestFunction()
        form = ngs.LinearForm(self._fes0)
        form += test * ngs.Norm(tangential) * ngs.ds(
            definedon=self.mesh.Boundaries(self.wall_boundary), skeleton=True)
        form.Assemble()

        integrated = np.asarray(form.vec).ravel()
        shear = np.zeros(self._fes0.ndof)
        shear[self._marked] = (integrated[self._marked]
                               / self._wall_measure[self._marked])
        shear = np.maximum(shear, 0.0)
        scale = max(1.0, float(np.max(shear[self._marked])))
        shear[shear < 1e-14 * scale] = 0.0
        return shear

    def _wall_viscosity_from_yplus(self, y_plus: np.ndarray) -> np.ndarray:
        """Log-law wall viscosity from resolved ``u_tau`` and cell distance."""
        wall_nu_t = np.zeros_like(y_plus)
        active = self.mask.vec.FV().NumPy() > 0.5
        log_cells = active & (y_plus > self.YPLUS_VISCOUS)
        if np.any(log_cells):
            u_plus = np.log(self.E_log * y_plus[log_cells]) / self.kappa
            wall_nu_t[log_cells] = self.nu * (
                y_plus[log_cells] / u_plus - 1.0)
        return np.maximum(wall_nu_t, 0.0)

    def update(self, K, U=None) -> None:
        """Refresh wall quantities using the configured friction-velocity method.

        Call once per Picard iteration, BEFORE ``eval_nu_t``.
        """
        u_tau = np.zeros(self._fes0.ndof)
        wall_nu_t = np.zeros(self._fes0.ndof)
        wall_shear = np.zeros(self._fes0.ndof)
        y_plus_cell = np.zeros(self._fes0.ndof)
        if self.u_tau_method == 0:
            projected = ngs.GridFunction(self._fes0)
            projected.Set(K)
            k_values = np.maximum(projected.vec.FV().NumPy(), 0.0)
            u_tau[self._marked] = self.C_mu ** 0.25 * np.sqrt(
                k_values[self._marked])
            y_plus_cell = (self._dist_cell.vec.FV().NumPy()
                           * u_tau / self.nu)
        else:
            if U is None:
                raise ValueError('Velocity-based u_tau requires the velocity iterate.')
            wall_shear = self._resolved_wall_shear(U)
            u_tau[self._marked] = np.sqrt(wall_shear[self._marked])
            y_plus_cell = (self._dist_cell.vec.FV().NumPy()
                           * u_tau / self.nu)
            wall_nu_t = self._wall_viscosity_from_yplus(y_plus_cell)

        wall_nu_t = self._wall_viscosity_from_yplus(y_plus_cell)

        self.u_tau_cell.vec.FV().NumPy()[:] = u_tau
        self.wall_nu_t_cell.vec.FV().NumPy()[:] = wall_nu_t
        self.wall_shear_cell.vec.FV().NumPy()[:] = wall_shear
        self._y_plus_cell_gf.vec.FV().NumPy()[:] = y_plus_cell
        self._warn_if_yplus_outside_recommended_range(y_plus_cell)

    def _warn_if_yplus_outside_recommended_range(self, y_plus: np.ndarray) -> None:
        """Warn once for each side of the recommended wall-function band."""
        wall_values = y_plus[self._marked]
        total = wall_values.size
        observed_min = float(np.min(wall_values))
        observed_max = float(np.max(wall_values))

        low_count = int(np.count_nonzero(
            wall_values < self.YPLUS_RECOMMENDED_MIN))
        if low_count and not self._warned_yplus_low:
            logging.warning(
                'Wall-function validity: %d/%d wall cells (%.1f%%) have y+ < %.0f; '
                'the equilibrium log-law treatment is recommended for %.0f <= y+ <= %.0f. '
                'Observed wall-cell range: %.6g <= y+ <= %.6g.',
                low_count, total, 100.0 * low_count / total,
                self.YPLUS_RECOMMENDED_MIN, self.YPLUS_RECOMMENDED_MIN,
                self.YPLUS_RECOMMENDED_MAX, observed_min, observed_max)
            self._warned_yplus_low = True

        high_count = int(np.count_nonzero(
            wall_values > self.YPLUS_RECOMMENDED_MAX))
        if high_count and not self._warned_yplus_high:
            logging.warning(
                'Wall-function validity: %d/%d wall cells (%.1f%%) have y+ > %.0f; '
                'the equilibrium log-law treatment is recommended for %.0f <= y+ <= %.0f. '
                'Observed wall-cell range: %.6g <= y+ <= %.6g.',
                high_count, total, 100.0 * high_count / total,
                self.YPLUS_RECOMMENDED_MAX, self.YPLUS_RECOMMENDED_MIN,
                self.YPLUS_RECOMMENDED_MAX, observed_min, observed_max)
            self._warned_yplus_high = True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def near_wall_mask(self) -> ngs.GridFunction:
        """1 only on cells that own a physical wall facet."""
        return self.mask

    def wall_facet_mask(self) -> ngs.GridFunction:
        """1 only on cells owning a physical wall facet."""
        return self.wall_facet_cell_mask

    def wall_distance_cell(self) -> ngs.GridFunction:
        return self._dist_cell

    def wall_distance_field(self) -> ngs.GridFunction:
        """Continuous Eikonal wall-distance field."""
        return self._dist_gf

    def y_plus_cell(self, K=None) -> ngs.CoefficientFunction:
        """Cellwise y+, from the cell-averaged wall distance."""
        if K is not None and self.u_tau_method == 0:
            self.update(K)
        return self._y_plus_cell_gf

    def y_plus_field(self, K=None) -> ngs.CoefficientFunction:
        """Pointwise y+, from the continuous Eikonal distance.

        Use this wherever y+ feeds a coefficient the weak form integrates over a
        volume, so the result varies across the wall cell instead of being one
        number per cell.
        """
        if K is not None and self.u_tau_method == 0:
            self.update(K)
        return self._dist_gf * self.u_tau_cell / self.nu

    def epsilon_wall_cell(self, K) -> ngs.CoefficientFunction:
        """Wall dissipation selected from one cell-averaged y+ value."""
        k_nonnegative = ngs.IfPos(K, K, 0.0)
        log_layer = (self.C_mu ** 0.75 * k_nonnegative ** 1.5
                     / (self.kappa * self._dist_cell))
        viscous_sublayer = (2.0 * self.nu * k_nonnegative
                            / self._dist_cell ** 2)
        return ngs.IfPos(self.y_plus_cell(K) - self.YPLUS_VISCOUS,
                         log_layer, viscous_sublayer)

    # ------------------------------------------------------------------
    # Eddy viscosity
    # ------------------------------------------------------------------

    def _wall_law(self, yplus, K, epsilon) -> ngs.CoefficientFunction:
        """Wall-only eddy viscosity as a function of the y+ handed in::

            nu_t = 0                                     y+ <  11.25   (sublayer)
            nu_t = nu*(y+ / ((1/kappa)*ln(E*y+)) - 1)     y+ >= 11.25   (log law)

        Wall-owner cells never fall back to the bulk k-epsilon viscosity.
        """
        # ln(E*y+) vanishes at y+ = 1/E; pinning y+ to YPLUS_VISCOUS below the
        # sublayer threshold avoids that root (the clamp below then zeroes it).
        yplus_log = ngs.IfPos(yplus - self.YPLUS_VISCOUS, yplus, self.YPLUS_VISCOUS)
        u_plus = ngs.log(self.E_log * yplus_log) / self.kappa
        log_law = self.nu * (yplus_log / u_plus - 1.0)
        log_law = ngs.IfPos(log_law, log_law, 0.0)

        return ngs.IfPos(yplus - self.YPLUS_VISCOUS, log_law, 0.0)

    def eval_nu_wall(self, K, epsilon) -> ngs.CoefficientFunction:
        """One wall eddy viscosity branch per wall cell, selected by P0 y+."""
        return self._wall_law(self.y_plus_cell(K), K, epsilon)

    def eval_nu_t(self, K, E) -> ngs.CoefficientFunction:
        """Wall law inside the mask, bulk ``C_mu k^2/epsilon`` outside it.

        Requires :meth:`update` to have run this iteration.
        """
        # Cell-based, not pointwise: the Eikonal distance is zero ON the wall, so
        # a pointwise wall law collapses there and jumps to bulk one cell in.
        nu_t_bulk = self.C_mu * K ** 2 / E
        nu_t = self.mask * self.eval_nu_wall(K, E) + (1 - self.mask) * nu_t_bulk
        return ngs.IfPos(nu_t, nu_t, 0)  # turbulent viscosity cannot be negative
