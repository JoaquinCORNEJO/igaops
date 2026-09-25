from typing import Callable, List, Optional, Union, Literal
from dataclasses import dataclass
from time import time
import logging

from scipy.sparse.linalg import LinearOperator, spsolve, lsqr
from scipy import sparse as sp
from numpy.linalg import norm
import numpy as np

from igaops.common import Constants, validate_entry
from .utils.convergence_manager import ConvergenceManager
from .utils.argsclass import SolverArgs
from .deflated.gcrodr import GCRODR

logger = logging.getLogger(__name__)


@dataclass
class OutputArgs:
    solution: np.ndarray
    residual: np.ndarray
    success: bool = False


class LinearSolver:
    """
    A class for solving linear systems of equations using various iterative methods.

    The class supports:
        Conjugate Gradient (CG),
        BiConjugate Gradient Stabilized (BiCGSTAB),
        Generalized Minimal Residual Method (GMRES),
        Block Conjugate Gradient.

    The class also provides direct solvers using sparse LU decomposition.

    Parameters
    ----------
    tolerance : float
        The convergence tolerance for the iterative solver.
    maxiters : int
        The maximum number of iterations for the iterative solver.
    cleandod : List[int], optional
        A list of indices to be masked (set to zero) in the residual vector during the iterative process. Default is ``None``.
    linear_type : str, optional
        The type of iterative solver to use. Must be one of ``cg``, ``block_cg``, ``bicgstab``, ``gcrodr``, or ``gmres``. Default is ``gmres``.
    verbose : bool, optional
        If True, prints convergence information and execution time. Default is ``True``.
    """

    def __init__(
        self,
        tolerance: float,
        maxiters: int,
        cleandod: Optional[List[int]] = None,
        linear_type: Literal["cg", "bicgstab", "gcrodr", "gmres", "block_cg"] = "gmres",
        verbose: bool = True,
    ):
        logger.debug("Initialize linear solver")

        self._config = SolverArgs(tolerance=tolerance, maxiters=maxiters)
        self.set_cleandod(cleandod)
        self._linear_type = validate_entry(
            linear_type, ["cg", "bicgstab", "gcrodr", "gmres", "block_cg"]
        )
        self._convergence_manager: Optional[ConvergenceManager] = None
        self._last_simulation: Optional[OutputArgs] = None
        self._verbose = verbose

        # For gcrodr we use a wrapper
        self._gcrodr_solver: Optional[GCRODR] = None

    @property
    def config(self):
        return self._config

    def set_cleandod(self, cleandod: Optional[List[int]]):
        """
        Set the indices to be masked (set to zero) in the residual vector during the iterative process.

        Parameters
        ----------
        cleandod : List[int], optional
            A list of indices to be masked. If None, no indices will be masked. Default is None.
        """
        if cleandod is not None and not isinstance(cleandod, list):
            raise TypeError("Cleandod should be list")
        self._cleandod = cleandod

    @property
    def linear_type(self):
        return self._linear_type

    @property
    def verbose(self):
        return self._verbose

    @property
    def last_simulation(self) -> OutputArgs:
        if self._last_simulation is None:
            return OutputArgs(
                solution=np.array([]), residual=np.array([]), success=False
            )
        return self._last_simulation

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

    def _apply_mask(self, x: np.ndarray):
        """
        Set to zero the values of x at indices in cleandod.
        """
        if self._cleandod is not None:
            x[self._cleandod] = 0.0

    def update(
        self,
        tolerance: Optional[float] = None,
        maxiters: Optional[float] = None,
    ):
        """
        Update the solver parameters.

        Parameters
        ----------
        tolerance : float, optional
            The new convergence tolerance for the iterative solver. If None, the existing value is retained.
        maxiters : int, optional
            The new maximum number of iterations for the iterative solver. If None, the existing value is retained.
        """
        kwargs = self.config.__dict__.copy()
        if tolerance is not None:
            kwargs["tolerance"] = tolerance
        if maxiters is not None:
            kwargs["maxiters"] = maxiters
        self._config = SolverArgs(**kwargs)
        self._convergence_manager = None

    def solve(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Optional[Callable] = None,
        x0: Optional[np.ndarray] = None,
        **mv_args,
    ) -> OutputArgs:
        """
        Solve the linear system Ax = b using the specified iterative method.

        Parameters
        -----------
        Afun : callable
            A function that computes the matrix-vector product A*x.
        b : array_like
            The right-hand side vector.
        Pfun : callable, optional
            A function that computes the preconditioner matrix-vector product P*x. Default is the identity function.
        x0 : array_like, optional
            A good guess to initialize algorithm.
        mv_args : dict, optional
            Additional arguments to pass to Afun (matrix-vector arguments).
            It can also be use to pass arguments to GCRODR solver. See notes.

        Returns
        -------
        OutputArgs
            A dataclass containing:
            'solution' (ndarray), the solution vector x;
            'residual' (ndarray), an array of relative residuals at each iteration;
            'success' (bool), if the solver has solved successfully or not.

        Notes
        -----
        GCRODR solver has the following extra arguments:
        - krylov_size: Maximum Krylov subspace dimension per cycle. Defaults to 40.
        - recycled_size: Number of approximate eigenvectors recycled across cycles / calls. Defaults to 20.
        - max_cycles: Fixed number of cycles (to avoid a forever while loop). Defaults to 10.
        """

        def identity(x: np.ndarray):
            return x

        self.convergence_manager.clear()
        Pfun = Pfun if callable(Pfun) else identity
        if not (callable(Afun) and callable(Pfun)):
            raise TypeError("Afun and Pfun should be callable")
        solver = {
            "cg": self._CG,
            "bicgstab": self._BICGSTAB,
            "gcrodr": self._gcrodr,
            "gmres": self._GMRES,
            "block_cg": self._BLOCK_CG,
        }[self.linear_type]

        # Get solver arguments (for instance only GCRODR)
        solver_args = {}
        for ky in ["krylov_size", "recycled_size", "max_cycles"]:
            if ky in mv_args:
                solver_args[ky] = mv_args[ky]
            mv_args.pop(ky, None)

        # Redefine functions
        def matvec(x: np.ndarray) -> np.ndarray:
            self._apply_mask(x)
            y = Afun(x, **mv_args)
            self._apply_mask(y)
            return y

        def precond(x: np.ndarray) -> np.ndarray:
            self._apply_mask(x)
            y = Pfun(x)
            self._apply_mask(y)
            return y

        # TODO: maybe use scipy iterative solver with callbacks
        start = time()

        b = np.copy(b)
        self._apply_mask(b)
        if x0 is not None:
            x0 = np.copy(x0)
            self._apply_mask(x0)
            b -= matvec(x0)
        output = solver(matvec, b, Pfun=precond, **solver_args)
        if x0 is not None:
            output.solution += x0

        self._last_simulation = output
        ndod = 0 if self._cleandod is None else len(self._cleandod)
        if self.verbose and self.linear_type != "gcrodr":
            stat = self.convergence_manager.get_status()
            it = stat["iteration"]["current"]
            it = -1 if it is None else it
            abserr = stat["absolute_error"]["current"] or 0.0
            relerr = stat["relative_error"]["current"] or 0.0
            relinc = stat["curr_increment"]["current"] or 0.0
            message = f"""
                Convergence summary in LINEAR SOLVER:
                - solve {len(b)} equations with {ndod} known variables
                - no. iterations {it}
                - abs residue {abserr:.2e}
                - rel residue {relerr:.2e}
                - rel increment {relinc:.2e}
                Execution time: {time() - start:.2e} seconds.
            """
            logger.info(message)
        return output

    def _CG(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Callable,
        **kwargs,
    ) -> OutputArgs:
        """
        Conjugate Gradient (CG) solver for solving the linear system Ax = b.
        """
        b = b.copy()
        x = np.zeros_like(b)
        r = b.copy()

        output = OutputArgs(solution=x, residual=np.array([1.0]), success=True)

        norm_0 = float(norm(r))
        self.convergence_manager.update(curr_absolute=norm_0)
        if self.convergence_manager.has_converged():
            logger.info("External force almost zero. No iterations")
            return output
        logger.debug(f"CG: Iteration {0} and residual {1.0}")

        ptilde = np.asarray(Pfun(r))
        p = np.copy(ptilde)
        rsold = r.dot(ptilde)
        res = [1.0]

        maxiters = int(self.convergence_manager.parameters["iteration"].reference)
        for it in range(1, maxiters + 1):
            Ap = np.asarray(Afun(p))
            alpha = rsold / Ap.dot(p)
            r -= alpha * Ap
            incr = alpha * p
            x += incr

            curr_incr = float(norm(incr) / (norm(x) + Constants.SAFEGUARD))
            norm_1 = float(norm(r))
            res.append(norm_1 / norm_0)
            logger.debug(f"CG: Iteration {it} and residual {res[-1]:.2e}")
            self.convergence_manager.update(
                curr_iteration=it,
                curr_absolute=norm_1,
                curr_relative=res[-1],
                curr_increment=curr_incr,
            )
            if self.convergence_manager.has_converged():
                break

            ptilde = np.asarray(Pfun(r))
            rsnew = ptilde.dot(r)
            p = ptilde + rsnew / rsold * p
            rsold = rsnew
        else:
            output.success = False
            it = self.convergence_manager.parameters["iteration"].current or 0
            logger.warning(f"No convergence after {it} iterations.")

        output.solution = x
        output.residual = np.asarray(res)
        return output

    def _BICGSTAB(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Callable,
        **kwargs,
    ) -> OutputArgs:
        """
        BiConjugate Gradient Stabilized (BiCGSTAB) method for solving linear systems.

        Notes
        ------
        This method iteratively solves the linear system A*x = b using the BiCGSTAB algorithm.
        The algorithm is suitable for large, sparse, and non-symmetric linear systems.
        """
        b = b.copy()
        x = np.zeros_like(b)
        r = b.copy()

        output = OutputArgs(solution=x, residual=np.array([1.0]), success=True)

        norm_0 = float(norm(r))
        self.convergence_manager.update(curr_absolute=norm_0)
        if self.convergence_manager.has_converged():
            logger.info("External force almost zero. No iterations")
            return output

        logger.debug(f"BICGSTAB: Iteration {0} and residual {1.0}")

        rhat = r.copy()
        p = r.copy()
        rsold = r.dot(rhat)
        res = [1.0]

        maxiters = int(self.convergence_manager.parameters["iteration"].reference)
        for it in range(1, maxiters + 1):
            ptilde = np.asarray(Pfun(p))
            Aptilde = np.asarray(Afun(ptilde))
            alpha = rsold / Aptilde.dot(rhat)
            s = r - alpha * Aptilde
            incr = alpha * ptilde
            x += incr

            curr_incr = float(norm(incr) / (norm(x) + Constants.SAFEGUARD))
            norm_1 = float(norm(s))
            res.append(norm_1 / norm_0)
            self.convergence_manager.update(
                curr_iteration=it,
                curr_absolute=norm_1,
                curr_relative=res[-1],
                curr_increment=curr_incr,
            )
            if self.convergence_manager.has_converged():
                break

            stilde = np.asarray(Pfun(s))
            Astilde = np.asarray(Afun(stilde))
            omega = Astilde.dot(s) / Astilde.dot(Astilde)
            r = s - omega * Astilde
            incr = omega * stilde
            x += incr

            curr_incr = float(norm(incr) / (norm(x) + Constants.SAFEGUARD))
            norm_1 = float(norm(r))
            res.append(norm_1 / norm_0)
            logger.debug(f"BICGSTAB: Iteration {it} and residual {res[-1]:.2e}")
            self.convergence_manager.update(
                curr_iteration=it,
                curr_absolute=norm_1,
                curr_relative=res[-1],
            )
            if self.convergence_manager.has_converged():
                break

            rsnew = rhat.dot(r)
            beta = (alpha / omega) * (rsnew / rsold)
            p = r + beta * (p - omega * Aptilde)
            rsold = np.copy(rsnew)
        else:
            output.success = False
            it = self.convergence_manager.parameters["iteration"].current or 0
            logger.warning(f"No convergence after {it} iterations.")

        output.solution = x
        output.residual = np.asarray(res)
        return output

    def _gcrodr(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Callable,
        **kwargs,
    ) -> OutputArgs:
        """
        Generalized Conjugate Residual with Deflated Restarting.
        """
        logger.warning(
            "GCRODR is experimental-only. Parametric study should be carried out to tune solver recycling."
        )
        if self._gcrodr_solver is None:
            krylov_size = kwargs.get("krylov_size", 40)
            recycled_size = kwargs.get("recycled_size", 20)
            max_cycles = kwargs.get("max_cycles", 10)
            self._gcrodr_solver = GCRODR(
                self.config.tolerance,
                self.config.maxiters,
                verbose=self.verbose,
                krylov_size=krylov_size,
                recycled_size=recycled_size,
                max_cycles=max_cycles,
            )
        output = self._gcrodr_solver.solve(Afun, b, Pfun)
        return OutputArgs(
            solution=output.solution, residual=output.residual, success=output.success
        )

    def _GMRES(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Callable,
        **kwargs,
    ) -> OutputArgs:
        """
        Generalized Minimal Residual Method (GMRES) for solving a linear system of equations.

        Notes
        -----
        This implementation uses the Arnoldi process to build an orthonormal basis
        for the Krylov subspace and solves the least squares problem to minimize the residual.
        """
        b = b.copy()
        x = np.zeros_like(b)
        r = b.copy()

        output = OutputArgs(solution=x, residual=np.array([1.0]), success=True)

        norm_0 = float(norm(r))
        self.convergence_manager.update(curr_absolute=norm_0)
        if self.convergence_manager.has_converged():
            logger.info("External force almost zero. No iterations")
            return output

        logger.debug(f"GMRES: Iteration {0} and residual {1.0}")
        maxiters = int(self.convergence_manager.parameters["iteration"].reference)

        Hessenberg = np.zeros((maxiters + 1, maxiters))
        AVectors = np.zeros((maxiters + 1, len(b)))
        PVectors = np.zeros_like(AVectors)

        AVectors[0] = r / norm_0
        y = np.zeros(1)
        res = [1.0]
        for it in range(1, maxiters + 1):
            p = np.asarray(Pfun(AVectors[it - 1]))
            PVectors[it - 1] = p

            w = np.asarray(Afun(p))

            for j in range(it):
                hij = w.dot(AVectors[j])
                Hessenberg[j, it - 1] += hij
                w -= hij * AVectors[j]

            # reorthogonalization pass
            for j in range(it):
                hij = w.dot(AVectors[j])
                Hessenberg[j, it - 1] += hij
                w -= hij * AVectors[j]

            norm_w = norm(w)
            Hessenberg[it, it - 1] = norm_w
            if norm_w != 0:
                AVectors[it] = w / norm_w

            mat = Hessenberg[: it + 1, :it]
            rhs = np.zeros(it + 1)
            rhs[0] = norm_0

            y = lsqr(mat, rhs, atol=Constants.TINY, btol=Constants.TINY)[0]
            xold = x.copy()
            x = PVectors[: len(y)].T @ y
            incr = x - xold

            curr_incr = float(norm(incr) / (norm(x) + Constants.SAFEGUARD))
            norm_1 = float(norm(rhs - mat @ y))
            res.append(norm_1 / norm_0)
            logger.debug(f"GMRES: Iteration {it} and residual {res[-1]:.2e}")
            self.convergence_manager.update(
                curr_iteration=it,
                curr_absolute=norm_1,
                curr_relative=res[-1],
                curr_increment=curr_incr,
            )
            if self.convergence_manager.has_converged():
                break
        else:
            output.success = False
            it = self.convergence_manager.parameters["iteration"].current or 0
            logger.warning(f"No convergence after {it} iterations.")

        output.solution = x
        output.residual = np.asarray(res)
        return output

    def _BLOCK_CG(
        self,
        Afun: Callable,
        b: np.ndarray,
        Pfun: Callable,
        **kwargs,
    ) -> OutputArgs:
        "Block Conjugate Gradient (Block CG) method for solving linear systems with multiple right-hand sides."
        if b.ndim != 2:
            raise ValueError("Expected 2D array.")

        b = b.copy()
        x = np.zeros_like(b)
        r = b.copy()

        output = OutputArgs(solution=x, residual=np.array([1.0]), success=True)

        norm_0 = float(norm(r))
        self.convergence_manager.update(curr_absolute=norm_0)
        if self.convergence_manager.has_converged():
            logger.info("External force almost zero. No iterations")
            return output

        logger.debug(f"BICGSTAB: Iteration {0} and residual {1.0}")

        z = np.asarray(Pfun(r))
        p = np.copy(z)
        res = [1.0]
        rtz_old = r.T @ z

        maxiters = int(self.convergence_manager.parameters["iteration"].reference)
        for it in range(1, maxiters + 1):
            Ap = np.asarray(Afun(p))
            PtAp = p.T @ Ap
            PtR = p.T @ r
            gamma = lsqr(PtAp, PtR, atol=Constants.TINY, btol=Constants.TINY)[0]

            r -= Ap @ gamma
            incr = p @ gamma
            x += incr

            curr_incr = float(norm(incr) / (norm(x) + Constants.SAFEGUARD))
            norm_1 = float(norm(r))
            res.append(norm_1 / norm_0)
            logger.debug(f"BLOCK CG: Iteration {it} and residual {res[-1]:.2e}")
            self.convergence_manager.update(
                curr_iteration=it,
                curr_absolute=norm_1,
                curr_relative=res[-1],
                curr_increment=curr_incr,
            )
            if self.convergence_manager.has_converged():
                break

            z = np.asarray(Pfun(r))
            rtz_new = r.T @ z
            delta = lsqr(rtz_old, rtz_new, atol=Constants.TINY, btol=Constants.TINY)[0]
            p = z + p @ delta
            rtz_old = rtz_new
        else:
            output.success = False
            it = self.convergence_manager.parameters["iteration"].current or 0
            logger.warning(f"No convergence after {it} iterations.")

        output.solution = x
        output.residual = np.asarray(res)
        return output

    @staticmethod
    def direct(
        A: Union[np.ndarray, sp.csr_array, LinearOperator],
        b: np.ndarray,
        verbose: bool = True,
    ):
        """
        Direct solver for linear systems using sparse LU decomposition.

        Parameters
        -----------
        A : array_like
            Coefficient matrix of the linear system.
        b : ndarray
            Right-hand side vector.
        verbose : Optional, bool
            If True, prints the time taken to solve the system. Default is True.

        Returns
        -------
        OutputArgs
            A dataclass containing:
            'solution' (ndarray), the solution vector x;
            'residual' (ndarray), an empty array (for consistency with iterative solvers).
        """
        start = time()
        if isinstance(A, LinearOperator):
            A = np.asarray(A @ np.eye(A.shape[1]))

        if isinstance(A, np.ndarray):
            A = sp.csr_array(A)
            A.eliminate_zeros()
        x = spsolve(A, b)
        if verbose:
            logger.info(f"Direct solver took {time() - start:.2e} seconds.")
        return OutputArgs(solution=np.asarray(x), residual=np.array([]), success=True)

    def __repr__(self) -> str:
        message = f""""
            \nLINEAR SOLVER:
            Max iterations: {self.config.maxiters}
            Tolerance: {self.config.tolerance}
            Linear type: {self.linear_type}
        """
        return message
