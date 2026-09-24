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

        RHO = self.lagrange_penalty
        NR = len(self.constraint_vector)

        if (not reuse or self._local_solver is None) and apply_P is not None:
            self._set_local_solver(apply_P, apply_Ptrans)

        def Hmatvec(xw: np.ndarray):
            "Computes [A, B^T; B, -I/rho]@[x; w]"
            x = xw[:-NR]
            w = xw[-NR:]
            mvup = apply_T(x) + self.dual_matrix.T @ w
            mvdw = self.constraint_matrix @ x - w / RHO
            return np.hstack((mvup, mvdw))

        def Hprecond(rxw: np.ndarray):
            """
            Computes [I, 0; BP^-1, I][P, 0; 0, S][I, P^-1B^T; 0, I]@[x; w]
            Here S is the Schur complement of Hmat, the exact value is
            S = -I/rho - B P^-1 B^T,
            but we replace it by a cheap approximation like -I/rho
            """
            if not callable(apply_P):
                return rxw

            rx = rxw[:-NR]
            rw = rxw[-NR:]

            # Solve [I, 0; BP^-1, I]
            dx1 = rx
            P_dx1 = apply_P(dx1)
            dw1 = rw - self.constraint_matrix @ P_dx1

            # Solve [P, 0; 0, S]
            dx2 = P_dx1
            dw2 = -self._get_tight_solution(
                self.local_solver.matvec, dw1, self.local_solver.preconditioner
            )

            # Solve [I, P^-1B^T; 0, I]
            dw = dw2
            dx = dx2 - apply_P(self.dual_matrix.T @ dw)

            return np.hstack((dx, dw))

        return self._get_loose_solution(Hmatvec, rhs, Hprecond)[:-NR]

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
        inner_residual = []

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
            new_rhs = np.hstack((rhs_x, np.zeros_like(dlam)))
            dx = self._solve_local_system(
                apply_T, new_rhs, apply_P, apply_Ptrans, reuse
            )
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
