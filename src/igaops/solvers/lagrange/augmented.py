from typing import Callable, Optional, Tuple
import logging

import numpy as np

from .alm_template import ALM_Template

logger = logging.getLogger(__name__)


class AugmentedLagrange(ALM_Template):
    """
    Class of Augmented Lagrangian method.
    """

    def _solve_local_system(
        self,
        apply_T: Callable,
        rhs: np.ndarray,
        apply_P: Optional[Callable],
        apply_Ptrans: Optional[Callable],
        reuse: bool,
    ):
        "Solve the system A + rho C C^T"

        if (not reuse or self._local_solver is None) and apply_P is not None:
            self._set_local_solver(apply_P, apply_Ptrans)

        def matvec(x: np.ndarray) -> np.ndarray:
            "Computes M11 x where M11 = (T + rho * C^T C)"
            CT_C_x = self.dual_matrix.T @ self.constraint_matrix @ x
            return apply_T(x) + self.lagrange_penalty * CT_C_x

        def preconditioner(x: np.ndarray):
            if not callable(apply_P):
                return x

            zz = apply_P(x)
            ss = np.asarray(self.constraint_matrix @ zz)
            ww = self.local_solver.solve(ss)
            tt = np.asarray(self.dual_matrix.T @ ww)
            uu = zz - apply_P(tt)
            return uu

        return self._get_loose_solution(matvec, rhs, preconditioner)

    def compute_increment_lag(
        self,
        apply_T: Callable,
        apply_P: Optional[Callable],
        lagrange_res: np.ndarray,
        current_sol: np.ndarray,
        apply_Ptrans: Optional[Callable] = None,
        reuse: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        RHO = self.lagrange_penalty
        NR = len(self.constraint_vector)
        residual = lagrange_res.copy()
        solution = current_sol.copy()

        def matvec_alm(z: np.ndarray) -> np.ndarray:
            "Computes y = [T x + C^T (rho C x + lam); C x]. Here z = (x, lam)"
            mvup = apply_T(z[:-NR])
            mvdw = self.constraint_matrix @ z[:-NR]
            mvup += self.dual_matrix.T @ (RHO * mvdw + z[-NR:])
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
            dx = self._solve_local_system(apply_T, rhs_x, apply_P, apply_Ptrans, reuse)
            self.inner_residual.append(self.loose_solver.last_simulation.residual)
            return np.hstack((dx, dlam))

        res_lam = self._compute_dual_residual(solution)
        res = np.hstack((residual, res_lam))
        delta = self._get_solution(matvec_alm, res, preconditioner_alm)

        if self.verbose:
            self.send_message()

        return delta[:-NR], delta[-NR:]
