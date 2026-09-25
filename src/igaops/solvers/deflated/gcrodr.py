from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import logging

from scipy.linalg import solve_triangular
from scipy.sparse.linalg import lsqr
from numpy.linalg import norm
import numpy as np

from igaops.common import Constants
from igaops.solvers.utils.argsclass import SolverArgs
from igaops.solvers.utils.convergence_manager import ConvergenceManager
from .helpers import gmres1, gmres2, getHarmVecs1, getHarmVecs2

logger = logging.getLogger(__name__)


@dataclass
class GcrodrArgs(SolverArgs):
    krylov_size: int
    recycled_size: int
    max_cycles: int

    def __post_init__(self):
        if self.krylov_size <= 0:
            raise ValueError("Krylov size should be positive")

        if self.recycled_size < 0:
            # NOTE: 0 means that gcrodr does not save any information
            raise ValueError("Recycle size should be greater than or equal to 0")

        if self.max_cycles <= 0:
            raise ValueError("Max no. cycles should greater than 0")

        if self.krylov_size < self.recycled_size:
            raise ValueError("Not consistent with the method")


@dataclass
class OutputArgs:
    solution: np.ndarray
    residual: np.ndarray
    success: bool = False


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------


class GCRODR:
    """
    GCRO with Deflated Restarting.

    Parameters
    ----------
    tolerance : (float)
        Relative residual tolerance.
    maxiters : int
        Total number of iterations (related to number of matvec).
    krylov_size : int
        Maximum Krylov subspace dimension per cycle.
    recycled_size : int
        Number of approximate eigenvectors recycled across cycles / calls.
    max_cycles : int
        Fixed number of cycles (to avoid a forever while loop).
    verbose : bool
        If yes, print information

    Notes
    -----
    Translated from the MATLAB implementation adapted from:
    Parks, Michael L., et al. "Recycling Krylov subspaces for sequences of
    linear systems." SIAM Journal on Scientific Computing 28.5 (2006): 1651-1674.
    """

    def __init__(
        self,
        tolerance: float,
        maxiters: int,
        krylov_size: int = 40,
        recycled_size: int = 20,
        max_cycles: int = 10,
        verbose: bool = True,
    ):
        self._verbose = verbose
        self._U_persist: Dict[str, np.ndarray] = {}
        self._config = GcrodrArgs(
            tolerance=tolerance,
            maxiters=maxiters,
            krylov_size=krylov_size,
            recycled_size=recycled_size,
            max_cycles=max_cycles,
        )
        self._convergence_manager: Optional[ConvergenceManager] = None

    @property
    def verbose(self):
        return self._verbose

    @property
    def convergence_manager(self) -> ConvergenceManager:
        if self._convergence_manager is None:
            cm = ConvergenceManager(
                refe_relative=self.config.tolerance,
                refe_iteration=self.config.maxiters,
                refe_absolute=Constants.SAFEGUARD,
            )
            cm.add_criterion("curr_increment", "relative_error")
            self._convergence_manager = cm
        return self._convergence_manager

    @property
    def config(self):
        return self._config

    def update(
        self,
        tolerance: Optional[float] = None,
        maxiters: Optional[float] = None,
    ):
        """
        Update the solver parameters. They are ignored if not defined.

        Parameters
        ----------
        tolerance : float, optional
            Linear solvers's tolerance. Should be a real in (0, 1).
        maxiters : int, optional
            Linear solvers's number of iterations. Should be positive integer.
        """
        kwargs = self.config.__dict__.copy()
        if tolerance is not None and np.isscalar(tolerance):
            kwargs["tolerance"] = float(np.abs(tolerance))
        if maxiters is not None and np.isscalar(maxiters):
            kwargs["maxiters"] = int(np.abs(maxiters))
        self._config = GcrodrArgs(**kwargs)
        self._convergence_manager = None

    def reset(self, reuse_name: str = "default"):
        """Discard the recycled subspace for the given key."""
        self._U_persist.pop(reuse_name, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_A(self, v):
        v = np.asarray(v)
        if np.iscomplexobj(v):
            vr = np.real(v)
            vi = np.imag(v)
            return self.A(vr) + 1j * self.A(vi)
        return self.A(v)

    def _apply_M(self, v):
        """Apply preconditioner  M v."""
        v = np.asarray(v)
        if np.iscomplexobj(v):
            vr = np.real(v)
            vi = np.imag(v)
            return self.M(vr) + 1j * self.M(vi)
        return self.M(v)

    def _abs_to_rel(self, array: Any):
        v = np.abs(array)
        v0 = max(v[0], Constants.SAFEGUARD)
        return v / v0

    def _send_message(self):
        if self.verbose:
            stat = self.convergence_manager.get_status()
            it = stat["iteration"]["current"]
            it = -1 if it is None else it
            abserr = stat["absolute_error"]["current"] or 0.0
            relerr = stat["relative_error"]["current"] or 0.0
            message = f"""
                Convergence summary in GCRODR:
                - nb. iterations {it},
                - abs residue {abserr:.2e}
                - rel residue {relerr:.2e}
            """
            logger.info(message)

    # ------------------------------------------------------------------
    # Public solver
    # ------------------------------------------------------------------

    def solve(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Optional[Callable] = None,
        reuse_name: str = "default",
    ) -> OutputArgs:
        """
        Solve  A x = b.

        Parameters
        ----------
        Afun : callable
            Matrix A as callable.
        b : ndarray
            Right-hand side vector.
        Pfun : callable, optional
            Preconditioner of matrix A. Defaults to None.
        reuse_name : str
            Key for the recycled subspace.

        Returns
        -------
        OutputArgs
            A dataclass containing:
            'solution' (ndarray), the solution vector x;
            'residual' (ndarray), an array of relative residuals at each iteration;
            'success' (bool), if the solver has solved successfully or not.
        """

        def identity(x: np.ndarray):
            return x

        if not callable(Afun):
            raise TypeError("Afun should be a function.")

        Pfun = Pfun if Pfun is not None else identity
        if not callable(Pfun):
            raise TypeError("Pfun should be a function.")

        self.convergence_manager.clear()

        # Save matvec operators
        self.A, self.M = Afun, Pfun

        # Recover information for algorithm
        m, k = self.config.krylov_size, self.config.recycled_size
        tolerance, max_cycles = self.config.tolerance, self.config.max_cycles

        # Initialize values
        dtype = np.complex128
        b = np.asarray(b, dtype=dtype)
        x = np.zeros_like(b, dtype=dtype)
        r = self._apply_M(b)
        norm_0 = float(norm(r))
        resvec = [norm_0]

        output = OutputArgs(
            solution=np.real(x), residual=self._abs_to_rel(resvec), success=True
        )

        self.convergence_manager.update(
            curr_absolute=norm_0,
            curr_iteration=0,
        )
        if self.convergence_manager.has_converged():
            logger.info("External force almost zero. No iterations")
            return output

        # ---- Initialise / recycle U ------------------------------------
        nmv = 0
        U = C = None
        if reuse_name in self._U_persist:
            # ---- Branch A: warm-start from a previous call -------------
            currcase = "warm start"
            Y = self._U_persist[reuse_name].copy()

            # C = M⁻¹ A U  (recompute; handles A varying between calls) -> n x k
            C = np.column_stack(
                [self._apply_M(self._apply_A(Y[:, i])) for i in range(Y.shape[1])]
            )
            # Orthonormalise C; adjust U so C = A U still holds: Q = A (U/R)
            C, R = np.linalg.qr(C, mode="reduced")
            U = solve_triangular(R.T, Y.T, lower=True).T

            Cr = C.conj().T.dot(r)
            x += U @ Cr
            r -= C @ Cr
            resvec.append(float(norm(r)))
            p = np.inf
        else:
            # ---- Branch B: prime the pump with one plain GMRES cycle ---
            currcase = "plain GMRES"
            x, r, V, H, p, rv = gmres1(
                self._apply_A, x, r, m, self._apply_M, tolerance * norm_0
            )
            resvec.extend(rv.tolist())
            nmv += p

            if p >= k:
                P = getHarmVecs1(p, k, H)
                Y = V[:p].T @ P
                Q, R = np.linalg.qr(H[: p + 1, :p] @ P, mode="reduced")
                C = V[: p + 1].T @ Q  # lift back to full space
                U = solve_triangular(R.T, Y.T, lower=True).T

        logger.info(f"GCRODR with {currcase}.")
        self.convergence_manager.update(
            curr_absolute=float(norm(r)),
            curr_relative=float(norm(r) / norm_0),
            curr_iteration=nmv,
        )
        if self.convergence_manager.has_converged() or p < m:
            logger.info(f"Convergence before cycling.")
            self._send_message()
            if U is not None:
                self._U_persist[reuse_name] = U
            output.solution = np.real(x)
            output.residual = self._abs_to_rel(resvec)
            return output

        # ---- Main GCRO-DR loop -----------------------------------------
        if U is None or C is None:
            raise RuntimeError("U and/or C are not initialized.")

        tol = Constants.SAFEGUARD
        currcycle = 0
        for currcycle in range(max_cycles):
            V, H_inner, B, p, rv = gmres2(
                self._apply_A, r, m - k, self._apply_M, C, tolerance * norm_0
            )
            resvec.extend(rv.tolist())
            nmv += p

            # Augmented Hessenberg  Gbar  of shape (p+k+1, p+k)
            #
            #   [ D     B[:, :p] ]   ← k rows
            #   [ 0     H[:p+1]  ]   ← p+1 rows
            #
            # Find D such as the columns of U @ D have unit norm
            D = np.diag(1.0 / np.sqrt(np.diag(U.conj().T @ U)))
            Utilde = U @ D
            Vhat = np.hstack([Utilde, V[:p].T])  # n × (p+k)
            What = np.hstack([C, V[: p + 1].T])  # n × (p+k+1)
            zeros = np.zeros((p + 1, k), dtype=dtype)
            Gbar = np.block(
                [
                    [D, B[:, :p]],
                    [zeros, H_inner[: p + 1, :p]],
                ]
            )

            # Solve minimization problem
            rhs = What.conj().T.dot(r)  # (p+k+1,)
            y = lsqr(Gbar, rhs, atol=tol, btol=tol)[0]

            # Update solution and residual
            x += Vhat @ y
            r -= What @ (Gbar @ y)

            self.convergence_manager.update(
                curr_absolute=float(norm(r)),
                curr_relative=float(norm(r) / norm_0),
                curr_iteration=nmv,
            )
            if self.convergence_manager.has_converged() or p < m - k:
                logger.info(f"Convergence during cycle {currcycle}.")
                self._send_message()
                break

            # Harmonic Ritz extraction for the next recycled subspace
            P = getHarmVecs2(p + k, k, Gbar, V[: p + 1], Utilde, C)

            Y = Vhat @ P
            Q, R = np.linalg.qr(Gbar @ P, mode="reduced")
            C = What @ Q
            U = solve_triangular(R.T, Y.T, lower=True).T

        else:
            output.success = False
            logger.warning(f"No convergence after {currcycle} cycles.")

        # ---- Converged -------------------------------------------------
        self._U_persist[reuse_name] = U
        output.solution = np.real(x)
        output.residual = self._abs_to_rel(resvec)
        return output
