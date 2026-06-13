from typing import Callable, List, Optional
from dataclasses import dataclass, field
from time import time
import logging

import numpy as np

from igaops.common import Constants
from .utils.argsclass import SolverArgs
from .utils.convergence_manager import ConvergenceManager
from .utils.inner_tolerance import InnerToleranceSetter
from .utils.update_manager import UpdateManager

logger = logging.getLogger(__name__)


@dataclass
class OutputArgs:
    success: bool = False
    nonlinear_residual: List[float] = field(default_factory=list)
    nonlinear_time: List[float] = field(default_factory=list)
    nonlinear_rate: List[float] = field(default_factory=list)
    linear_tolerance: List[float] = field(default_factory=list)


@dataclass
class NonLinearArgs(SolverArgs):
    allow_acceleration: bool
    allow_line_search: bool
    linesearch_maxiters: int = 4
    linesearch_miniter: int = 0
    anderson_maxsize: int = 4
    anderson_miniter: int = 1

    def __post_init__(self):
        """Validate arguments after initialization."""
        if not isinstance(self.allow_acceleration, bool):
            raise TypeError(
                f"It should be boolean, got {type(self.allow_acceleration)}"
            )

        if not isinstance(self.allow_line_search, bool):
            raise TypeError(f"It should be boolean, got {type(self.allow_line_search)}")

        # Other requirements
        assert self.linesearch_maxiters > 0
        assert self.linesearch_miniter >= 0
        assert self.anderson_maxsize > 0
        assert self.anderson_miniter >= 0
        assert self.anderson_miniter < self.anderson_maxsize


class NonLinearSolver:
    """
    A class for solving nonlinear problems of the form res(x) = 0 using iterative methods.
    The solver supports Anderson acceleration and backtracking line search to enhance convergence.

    Parameters
    ----------
    tolerance : float
        Convergence tolerance for the nonlinear solver
    maxiters : int
        Maximum number of iterations allowed
    allow_acceleration : float
        Whether to use Anderson acceleration
    allow_line_search : float
        Whether to use backtracking line search
    verbose : bool
        Whether to print convergence information during iterations. Default is ``True``.
    update_manager : UpdateManager, optional
        Custom update manager instance
    inner_tolerance_setter : InnerToleranceSetter, optional
        Custom inner tolerance setter instance
    """

    def __init__(
        self,
        tolerance: float,
        maxiters: int,
        allow_acceleration: bool,
        allow_line_search: bool,
        verbose: bool = True,
        update_manager: Optional[UpdateManager] = None,
        inner_tolerance_setter: Optional[InnerToleranceSetter] = None,
    ):
        logger.debug("Initialize nonlinear solver")
        # Internal variables
        self._config = NonLinearArgs(
            tolerance=tolerance,
            maxiters=maxiters,
            allow_acceleration=allow_acceleration,
            allow_line_search=allow_line_search,
        )
        self._verbose = verbose

        self._convergence_manager: Optional[ConvergenceManager] = None

        self._update_manager: Optional[UpdateManager] = None
        if isinstance(update_manager, UpdateManager):
            self._update_manager = update_manager

        self._inner_tolerance_setter: Optional[InnerToleranceSetter] = None
        if isinstance(inner_tolerance_setter, InnerToleranceSetter):
            self._inner_tolerance_setter = inner_tolerance_setter

        self._last_residual_output: dict = {}

    @property
    def config(self) -> NonLinearArgs:
        return self._config

    @property
    def update_manager(self):
        if not isinstance(self._update_manager, UpdateManager):
            self._update_manager = UpdateManager()
        return self._update_manager

    @property
    def inner_tolerance_setter(self) -> InnerToleranceSetter:
        if not isinstance(self._inner_tolerance_setter, InnerToleranceSetter):
            inner_tolerance = InnerToleranceSetter(
                inner_tolerance_type="exact_picard",
                inner_tolerance_args={"default": Constants.TINY},
            )
            self._inner_tolerance_setter = inner_tolerance
        return self._inner_tolerance_setter

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
    def last_residual_output(self) -> dict:
        return self._last_residual_output

    def update(
        self,
        tolerance: Optional[float] = None,
        maxiters: Optional[float] = None,
        allow_acceleration: Optional[bool] = None,
        allow_line_search: Optional[bool] = None,
    ):
        """
        Update solver parameters and internal managers.

        Parameters
        ----------
        tolerance : float, optional
            New convergence tolerance
        maxiters : int, optional
            New maximum number of iterations
        allow_acceleration : bool, optional
            Whether to use Anderson acceleration
        allow_line_search : bool, optional
            Whether to use backtracking line search
        """
        kwargs = self._config.__dict__.copy()
        if tolerance is not None:
            kwargs["tolerance"] = tolerance
        if maxiters is not None:
            kwargs["maxiters"] = maxiters
        if allow_acceleration is not None:
            kwargs["allow_acceleration"] = allow_acceleration
        if allow_line_search is not None:
            kwargs["allow_line_search"] = allow_line_search
        self._config = NonLinearArgs(**kwargs)
        self._convergence_manager = None

    def _anderson_acceleration(
        self, incr_hist: List[np.ndarray], sol_hist: List[np.ndarray]
    ) -> np.ndarray:
        """
        Anderson acceleration for fixed-point iterations.

        The problem to be solved is A(u) * u = b
        The fixed point is A(u_k) * u_{k+1} = b, with u_0 = 0
        The problem can be also written as Newton: u_{k+1} = u_k + incr_k
        where incr_k = (A(u_k))^{-1} @ res_k and res_k = b - A(u_k) * u_k

        For Anderson acceleration we define f_k = (A(u_k))^{-1} @ b,
        and g_k = f_k - u_k = (A(u_k))^{-1} @ b - u_k = (A(u_k))^{-1} @ res_k = incr_k

        Let define G = [g_{k-m+1} - g_{k-m}, ..., g_k - g_{k-1}], then we compute
        the solution of min || G @ alpha - g_k||^2 over alpha in R^m using QR decomposition
        Finally, let define X = [u_{k-m+1} - u_{k-m}, ..., u_k - u_{k-1}]
        the new increment is g_k - alpha @ (G + X)

        Parameters
        ----------
        incr_hist : list
            History of increments g_i for i = k-m, ..., k
        sol_hist : list
            History of solutions u_i for i = k-m, ..., k

        Returns
        -------
        ndarray
            The Anderson accelerated solution.
        """

        logger.debug("Starting fixed-point acceleration with Anderson algorithm")

        # Number of previous directions available
        m = len(incr_hist) - 1
        if m < self.config.anderson_miniter:
            # Not enough history, fallback to fixed point
            return incr_hist[-1]

        # Stack g_k into matrix G (ndof × m))
        G = np.column_stack([incr_hist[i + 1] - incr_hist[i] for i in range(m)])
        X = np.column_stack([sol_hist[i + 1] - sol_hist[i] for i in range(m)])

        # Right-hand side is current increment g_k
        g_k = incr_hist[m]

        # Solve least squares problem: min ||G * gamma - g_k||
        alpha = np.linalg.lstsq(G, g_k, rcond=None)[0]

        if np.linalg.norm(alpha) > Constants.HUGE:
            # Value too high fallback to fixed point
            return incr_hist[-1]

        # Compute accelerated increment
        return g_k - (G + X) @ alpha

    def _backtracking_line_search(
        self,
        sol_current: np.ndarray,
        incr_current: np.ndarray,
        compute_residual: Callable,
        **residual_args,
    ) -> float:
        """
        Backtracking line search using the Armijo condition.

        The merit function is:
            phi(x) = 0.5 * ||R(x)||^2.

        For a Newton step, the directional derivative of phi at alpha=0 can be approximated as:
            phi'(0) ≈ -2 * phi(0).

        Parameters
        ----------
        sol_current : ndarray
            Current solution vector.
        incr_current : ndarray
            Newton increment (search direction).
        compute_residual : Callable
            Function that returns the residual vector.
        residual_args : dict
            Additional arguments passed to compute_residual.

        Returns
        -------
        float
            Step length alpha.
        """

        # 1. Line search parameters
        alpha = 1.0  # Initial full Newton step
        rho = 0.5  # Step reduction factor
        c1 = 1e-4  # Armijo sufficient decrease parameter
        alpha_min = Constants.TINY  # Minimum allowed step size

        logger.debug("Starting Backtracking Line Search (Armijo)")

        # 2. Evaluate residual at current solution
        res_0 = compute_residual(sol_current, **residual_args)[0]
        phi_0 = 0.5 * np.linalg.norm(res_0) ** 2

        # Directional derivative approximation for Newton methods
        # This assumes a reasonably accurate Newton direction
        phi_prime_0 = -2.0 * phi_0

        for iteration in range(self.config.linesearch_maxiters):

            # 3. Trial step
            sol_trial = sol_current + alpha * incr_current

            # 4. Evaluate residual at trial solution
            res_trial = compute_residual(sol_trial, **residual_args)[0]
            phi_trial = 0.5 * np.linalg.norm(res_trial) ** 2

            # 5. Armijo condition
            if phi_trial <= phi_0 + c1 * alpha * phi_prime_0:
                logger.debug(
                    "Line search accepted: alpha=%.3e, phi=%.3e, reductions=%d",
                    alpha,
                    phi_trial,
                    iteration,
                )
                return float(alpha)

            # 6. Reduce step length
            alpha *= rho

            logger.debug(
                "Line search reduction: alpha=%.3e, phi_trial=%.3e", alpha, phi_trial
            )

            # 7. Safeguard against excessively small steps
            if alpha < alpha_min:
                logger.debug(
                    "Line search reached minimum step size (alpha=%.3e). "
                    "Returning smallest step.",
                    alpha,
                )
                return float(alpha)

        logger.debug(
            "Line search reached maximum iterations. Returning alpha=%.3e", alpha
        )
        return float(alpha)

    def solve(
        self,
        solution: np.ndarray,
        compute_residual: Callable,
        compute_increment: Callable,
        residual_args: Optional[dict] = None,
        increment_args: Optional[dict] = None,
    ) -> OutputArgs:
        """
        Solve the nonlinear problem res(x) = 0 using an iterative method.

        Parameters
        ----------
        solution : ndarray
            Initial guess for the solution, updated in-place.
        compute_residual : Callable
            Function that computes the residual vector given a solution.
        compute_increment : Callable
            Function that computes the increment (search direction) given a residual.
        residual_args : dict, optional
            Additional arguments to pass to compute_residual.
        increment_args : dict, optional
            Additional arguments to pass to compute_increment.

        Returns
        -------
        OutputArgs
            A class containing convergence history and extra data from nonlinear evaluations.
        """

        if not (callable(compute_residual) and callable(compute_increment)):
            raise TypeError("compute_residual and compute_increment must be callable.")

        if residual_args is None:
            residual_args = {}

        if increment_args is None:
            increment_args = {}

        # Clear current values of Convergence
        self.convergence_manager.clear()
        self.update_manager.reset_newton()

        # Variables for inner and outer tolerance
        norm_increment, norm_solution = 1.0, 1.0  # Dummy initialization
        norm_residual_old = None  # Dummy initialization
        inner_tolerance_old = None  # Dummy initialization
        norm_residual_ref = 0.0  # Dummy initialization

        # Variable for Anderson acceleration
        incr_history: List[np.ndarray] = []
        solu_history: List[np.ndarray] = []

        # Save data
        output = OutputArgs()

        start = time()
        if self._verbose:
            logger.info(f"ITERATION | ABS RESIDUAL")
            logger.info("=" * 30)

        for it in range(self.config.maxiters):
            # Compute residual
            residual, info_to_save = compute_residual(solution, **residual_args)
            norm_residual = float(np.linalg.norm(residual))
            logger.debug(f"Iteration {it} and residual {norm_residual:.2e}")
            self._last_residual_output = info_to_save

            if self._verbose:
                logger.info(f" IT {it} : ABS {norm_residual:.2e}")

            # Update convergence manager
            if it == 0:
                norm_residual_ref = norm_residual + Constants.SAFEGUARD

            curr_relative = float(norm_residual / norm_residual_ref)
            curr_increment = float(norm_increment / norm_solution)
            self.convergence_manager.update(
                curr_iteration=it,
                curr_absolute=norm_residual,
                curr_relative=curr_relative,
                curr_increment=curr_increment,
            )

            # Save data
            output.nonlinear_residual.append(norm_residual)
            output.nonlinear_time.append(time() - start)
            output.nonlinear_rate.append(curr_increment)

            if self.convergence_manager.has_converged():
                output.success = True
                break

            # Compute internal tolerance
            inner_tolerance = self.inner_tolerance_setter.compute(
                norm_residual, norm_residual_old, inner_tolerance_old
            )
            norm_residual_old = norm_residual
            inner_tolerance_old = inner_tolerance
            output.linear_tolerance.append(inner_tolerance)

            # Compute increment
            increment_args["inner_tolerance"] = inner_tolerance
            increment_args["current_solution"] = solution
            increment: np.ndarray = compute_increment(residual, **increment_args)

            # Compute Anderson acceleration
            if self.config.allow_acceleration:
                if it >= self.config.anderson_maxsize:
                    incr_history.pop(0)
                    solu_history.pop(0)
                incr_history.append(increment.copy())
                solu_history.append(solution.copy())
                increment = self._anderson_acceleration(incr_history, solu_history)

            # Compute line-search
            if self.config.allow_line_search and it > self.config.linesearch_miniter:
                alpha = self._backtracking_line_search(
                    solution, increment, compute_residual, **residual_args
                )
                increment *= alpha

            # Update active control points
            solution += increment

            # Compute norms for outer stopping criterion
            norm_increment = np.linalg.norm(increment)
            norm_solution = np.linalg.norm(solution) + Constants.SAFEGUARD

            # Update manager
            self.update_manager.increment_newton()
        else:
            output.success = False
            it = self.convergence_manager.parameters["iteration"].current or 0
            logger.warning(f"No convergence after {it + 1} iterations.")

        if self._verbose:
            stat = self.convergence_manager.get_status()
            it = stat["iteration"]["current"]
            it = -1 if it is None else it
            abserr = stat["absolute_error"]["current"] or 0.0
            relerr = stat["relative_error"]["current"] or 0.0
            relinc = stat["curr_increment"]["current"] or 0.0
            message = f"""
                Convergence summary in NONLINEAR SOLVER:
                - solution's shape is {solution.shape}
                - no. iterations {it}
                - abs residue {abserr:.2e}
                - rel residue {relerr:.2e}
                - rel increment {relinc:.2e}
                Execution time: {time() - start:.2e} seconds.
            """
            logger.info(message)

        return output

    def __repr__(self) -> str:
        message = f""""
            \nNON LINEAR SOLVER:
            Max iterations: {self.config.maxiters}
            Tolerance: {self.config.tolerance}
            With Anderson acceleration: {self.config.allow_acceleration}
            With line search: {self.config.allow_line_search}
        """
        return message
