from typing import Callable, Optional
import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator

from igaops.solvers.linear_solver import LinearSolver

from .schur_solver import SchurSolver
from .template import Template

logger = logging.getLogger(__name__)


class ALM_Template(Template):
    """
    Class of Augmented Lagrangian method.
    """

    # Choose a very loose tolerance and few iterations
    # Eventually we can modify this choice in children
    loose_solver = LinearSolver(
        tolerance=1e-3,
        maxiters=25,
        linear_type="gmres",
        verbose=False,
    )

    tight_solver = LinearSolver(
        tolerance=1e-8,
        maxiters=100,
        linear_type="gmres",
        verbose=False,
    )

    @property
    def local_solver(self) -> SchurSolver:
        if self._local_solver is None:
            raise RuntimeError("Local solver is undefined")
        return self._local_solver

    @property
    def is_penalty_necessary(self):
        return True

    def _get_loose_solution(self, apply_M: Callable, x: np.ndarray, apply_P: Callable):
        return self.loose_solver.solve(apply_M, x, apply_P).solution

    def _get_tight_solution(self, apply_M: Callable, x: np.ndarray, apply_P: Callable):
        return self.tight_solver.solve(apply_M, x, apply_P).solution

    def _set_local_solver(self, apply_P: Callable, apply_Ptrans: Optional[Callable]):
        n = len(self.constraint_vector)
        mu = 1.0 / self.lagrange_penalty
        args = dict(
            matvec=lambda x: self.constraint_matrix @ apply_P(self.dual_matrix.T @ x)
        )
        if callable(apply_Ptrans):
            args.update(
                rmatvec=lambda x: self.dual_matrix
                @ apply_Ptrans(self.constraint_matrix.T @ x)
            )
        A = LinearOperator(dtype=float, shape=(n, n), **args)
        self._local_solver = SchurSolver(A=A, mu=mu, solver_type="lu")

    def compute_residual_lag(
        self,
        current_res: np.ndarray,
        lagrange_multiplier: np.ndarray,
        current_sol: Optional[np.ndarray],
    ):
        if not isinstance(current_sol, np.ndarray):
            raise TypeError("Current solution should be array in ALM.")

        # Current residual =  f - A @ U
        updated_res = current_res.copy()

        # For Lagrange residual f - A @ U - C^T @ lambda + rho C^T @ (g - C u)
        updated_res -= self.dual_matrix.T @ lagrange_multiplier

        rho = self.lagrange_penalty
        res_lam = self._compute_dual_residual(current_sol)
        updated_res += rho * (self.dual_matrix.T @ res_lam)

        return updated_res
