from typing import Callable, Optional, Tuple, Any
import logging

import numpy as np

from .alm_template import ALM_Template, preconditioner_like

logger = logging.getLogger(__name__)


class AugmentedLagrange(ALM_Template):
    """
    Class of Augmented Lagrangian method.
    """

    def _solve_local_system(
        self,
        apply_T: Callable,
        rhs: np.ndarray,
        preconditioner: Optional[preconditioner_like],
        reuse: bool,
    ):
        """
        Solve the system A_aug x = (A + rho B B^T) x = r_x as a linear system
        [A      B  ][x] = [rx]
        [B^T -I/rho][w]   [0 ]
        The preconditioner to this new system is:
        [I      0][P 0][I P^-1B^T]
        [BP^-1  I][0 S][0    I   ]
        where the Schur complement S = -(I/rho + B P^-1 B^T)
        """

        RHO = self.lagrange_penalty
        NR = len(self.constraint_vector)

        if (not reuse or self._local_solver is None) and preconditioner is not None:
            self._set_local_solver(preconditioner)

        def new_matvec(xw: np.ndarray):
            "Computes matvec product with [A, B^T; B, -I/rho]"
            x = xw[:-NR]
            w = xw[-NR:]
            mvup = apply_T(x) + self.dual_matrix.T @ w
            mvdw = self.constraint_matrix @ x - w / RHO
            return np.hstack((mvup, mvdw))

        def new_precond(rxw: np.ndarray):
            """
            Computes the matvec product with
            [I, 0; BP^-1, I][P, 0; 0, S][I, P^-1B^T; 0, I].
            Here S is the Schur complement written like:
            S = -I/rho - B P^-1 B^T.
            """
            if preconditioner is None:
                return rxw

            rx = rxw[:-NR]
            rw = rxw[-NR:]

            # Solve [I, 0; BP^-1, I]
            dx1 = rx
            P_dx1 = preconditioner.apply(dx1)
            dw1 = rw - self.constraint_matrix @ P_dx1

            # Solve [P, 0; 0, S]
            dx2 = P_dx1
            dw2 = -self.local_solver.solve(dw1)

            # Solve [I, P^-1B^T; 0, I]
            dw = dw2
            dx = dx2 - preconditioner.apply(self.dual_matrix.T @ dw)

            return np.hstack((dx, dw))

        new_rhs = np.hstack((rhs, np.zeros(NR)))
        return self._get_loose_solution(new_matvec, new_rhs, new_precond)[:-NR]

    def compute_increment_lag(
        self,
        apply_T: Callable,
        preconditioner: Optional[Any],
        lagrange_res: np.ndarray,
        current_sol: np.ndarray,
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
            if preconditioner is None:
                return rz

            dlam = -RHO * rz[-NR:]
            rhs_x = rz[:-NR] - self.dual_matrix.T @ dlam
            dx = self._solve_local_system(apply_T, rhs_x, preconditioner, reuse)
            self.inner_residual.append(self.loose_solver.last_simulation.residual)
            return np.hstack((dx, dlam))

        res_lam = self._compute_dual_residual(solution)
        res = np.hstack((residual, res_lam))
        delta = self._get_solution(matvec_alm, res, preconditioner_alm)

        if self.verbose:
            self.send_message()

        return delta[:-NR], delta[-NR:]
