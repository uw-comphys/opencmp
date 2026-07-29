"""
Nonlinear mixing strategies for the stationary inexact Newton iteration in base_solver._solve().

Each mixer owns the per-iteration state for its scheme (alpha, history vectors, etc.) and
exposes a single step(f_vec, x_prev_vec, num_iterations) -> np.ndarray call.  The loop in
_solve() is responsible for everything else (BC application, re-assembly, convergence checks,
gfu/x_curr updates).

All three schemes follow the same first-step convention as the original code: on iteration 2
(the first real mixing step) every scheme performs linear mixing and initialises alpha from
the dominant-eigenvalue estimate.  Scheme-specific logic takes over from iteration 3 onward.
"""

import numpy as np
import numpy.linalg as npl
from ngsolve import BaseVector


class LinearMixing:
    """dx = alpha * f  (constant Jacobian approximation J ~ -1/alpha)."""

    def __init__(self, alpha: float = 1.0, **_) -> None:
        self._alpha = alpha

    def step(self, f_vec: BaseVector, x_prev_vec: BaseVector, num_iterations: int) -> np.ndarray:
        if num_iterations == 2:
            # TODO: must handle this after implementing alpha into config file and finding
            # a citation or reference for this estimate and ITS BOUNDS
            # there is no citation for this from the Scipy linear_mixing solver code, but it seems to be based
            # on the "dominent eigenvalue method" (reference needed!)
            # based on Scipy implementation and information from the DEM method, alpha should not exceed 1
            # for stability.
            self._alpha = min(1., 0.5 * max(1., x_prev_vec.Norm()) / f_vec.Norm())
        return self._alpha * f_vec.FV().NumPy().copy()


class DiagBroyden:
    """Diagonal Broyden update: beta is a vector-valued inverse-Jacobian diagonal."""

    def __init__(self, alpha: float = 1.0, **_) -> None:
        self._alpha = alpha
        self._fprev: np.ndarray = None
        self._beta: np.ndarray = None
        self._dx_prev_np: np.ndarray = None

    def step(self, f_vec: BaseVector, x_prev_vec: BaseVector, num_iterations: int) -> np.ndarray:
        fcurr = f_vec.FV().NumPy().copy()
        if num_iterations == 2:
            self._alpha = min(1., 0.5 * max(1., x_prev_vec.Norm()) / f_vec.Norm())
            self._fprev = fcurr.copy()
            # TODO: needs to be made thread efficient; currently repeated by every thread
            self._beta = np.full(f_vec.size, 1.0 / self._alpha)
            dx_np = self._alpha * fcurr
        else:
            # Jacobian update
            dx_prev = self._dx_prev_np
            self._beta -= (fcurr - self._fprev + self._beta * dx_prev) * dx_prev / npl.norm(dx_prev) ** 2
            dx_np = fcurr / self._beta
            self._fprev = fcurr.copy()
        self._dx_prev_np = dx_np.copy()
        return dx_np


class Anderson:
    """
    Anderson mixing (Eyert, J. Comp. Phys. 124, 271 (1996)).

    Retains keep_vectors difference vectors (dx_all, df_all) and constructs a
    weighted combination to accelerate convergence.  Falls back to linear mixing
    when the history matrix A is singular.
    """

    def __init__(self, alpha: float = 1.0, keep_vectors: int = 5,
                 w0: float = 0.01, singular_tolerance: float = 1e-6) -> None:
        self._alpha = alpha
        self._keep_vectors = keep_vectors
        self._w0 = w0
        self._singular_tolerance = singular_tolerance
        self._fprev: np.ndarray = None
        self._dx_all: list = []
        self._df_all: list = []

    def step(self, f_vec: BaseVector, x_prev_vec: BaseVector, num_iterations: int) -> np.ndarray:
        fcurr = f_vec.FV().NumPy().copy()

        if num_iterations == 2:
            self._alpha = min(1., 0.5 * max(1., x_prev_vec.Norm()) / f_vec.Norm())
            self._fprev = fcurr.copy()
            dx_np = self._alpha * fcurr
            self._dx_all = [dx_np.copy()]
            self._df_all = []
            return dx_np

        # num_iterations > 2: full Anderson mixing
        self._df_all.append(fcurr - self._fprev)
        self._fprev = fcurr.copy()

        if len(self._dx_all) > self._keep_vectors:
            self._dx_all.pop(0)
            self._df_all.pop(0)

        A = np.zeros((len(self._dx_all), len(self._dx_all)))
        for i in range(len(self._dx_all)):
            for j in range(i, len(self._dx_all)):
                A[i, j] = np.vdot(self._df_all[i], self._df_all[j])
        np.fill_diagonal(A, A.diagonal() * (1 + self._w0 ** 2))
        A += np.triu(A, 1).T.conj()

        if abs(npl.det(A)) < self._singular_tolerance:
            # reset jacobian approximation
            self._dx_all = []
            self._df_all = []
            dx_np = self._alpha * fcurr
            self._dx_all.append(dx_np.copy())
        else:
            dx_np = -self._alpha * fcurr
            dff = np.array([np.vdot(self._df_all[k], fcurr) for k in range(len(self._df_all))])
            gamma = npl.solve(A, dff)
            for k in range(len(self._df_all)):
                dx_np += gamma[k] * (self._dx_all[k] + self._alpha * self._df_all[k])
            dx_np = -1 * dx_np
            self._dx_all.append(dx_np.copy())

        return dx_np


_SCHEMES = {
    'LinearMixing': LinearMixing,
    'DiagBroyden':  DiagBroyden,
    'Anderson':     Anderson,
    'default':      Anderson,
}


def make_mixer(name: str, alpha: float = 1.0, keep_vectors: int = 5,
               w0: float = 0.01, singular_tolerance: float = 1e-6):
    """Return the mixer instance for the given nonlinear_solver config value."""
    if name not in _SCHEMES:
        raise ValueError(f"Unknown nonlinear_solver '{name}'. Choose from: {list(_SCHEMES)}")
    return _SCHEMES[name](alpha=alpha, keep_vectors=keep_vectors,
                          w0=w0, singular_tolerance=singular_tolerance)
