from typing import Callable, Optional, Tuple
import logging

import numpy as np

from igaops.solvers.linear_solver import LinearSolver
from .template import Template

logger = logging.getLogger(__name__)


class AugmentedLagrange(Template):
    """
    Class of Augmented Lagrangian method.
    """

    loose_solver = LinearSolver(
        tolerance=1e-3,  # Choose a very loose tolerance
        maxiters=25,
        linear_type="gmres",
        verbose=False,
    )

    @property
    def is_penalty_necessary(self):
        return True

    def _get_loose_solution(self, apply_M: Callable, x: np.ndarray, apply_P: Callable):
        return self.loose_solver.solve(apply_M, x, apply_P).solution

    def compute_increment_lag(
        self,
        apply_T: Callable,
        apply_P: Optional[Callable],
        lagrange_res: np.ndarray,
        current_sol: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        RHO = self.lagrange_penalty
        NR = len(self.constraint_vector)
        residual = lagrange_res.copy()
        solution = current_sol.copy()
        inner_residual = []

        def matvec_11(x: np.ndarray) -> np.ndarray:
            "Computes M11 x where M11 = (T + rho * C^T C)"
            Tv = apply_T(x)
            Cv = self.constraint_matrix @ x
            CT_C_v = self.dual_matrix.T @ Cv
            return Tv + RHO * CT_C_v

        def matvec_alm(z: np.ndarray) -> np.ndarray:
            "Computes y = [M11 x + C^T lam; C x]. Here z = (x, lam)"
            mvup = matvec_11(z[:-NR]) + self.dual_matrix.T @ z[-NR:]
            mvdw = self.constraint_matrix @ z[:-NR]
            return np.hstack((mvup, mvdw))

        def preconditioner_alm(rz: np.ndarray) -> np.ndarray:
            """
            Apply a preconditioner for the ALM matrix: [M11, C^T; 0, -I/rho]
            Here M11 is solved with a loose tolerance using preconditioner P.
            """
            if not callable(apply_P):
                return rz

            dlam = -RHO * rz[-NR:]
            rhs_x = rz[:-NR] - self.dual_matrix.T @ dlam
            dx = self._get_loose_solution(matvec_11, rhs_x, apply_P)
            inner_residual.append(self.loose_solver.last_simulation.residual)
            return np.hstack((dx, dlam))

        res_lam = self._compute_dual_residual(solution)
        res = np.hstack((residual, res_lam))
        delta = self._get_solution(matvec_alm, res, preconditioner_alm)

        if self.verbose:
            logger.info(f"Loose solver info: {self.loose_solver}")
            message = "\nIT. | NO. MATVEC | LAST RESIDUAL"
            for i, x in enumerate(inner_residual):
                message += f"\n{i}\t {len(x)}\t {x[-1]:.1e}\n"
            logger.info(f"Residuals of loose solver: {message}")

        return delta[:-NR], delta[-NR:]

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
