from typing import Callable, Union, Optional, Literal, Tuple, Any
from abc import ABC, abstractmethod
import logging

from scipy.sparse.linalg import LinearOperator
from scipy import sparse as sp
import numpy as np

from igaops.common import Constants
from igaops.solvers.linear_solver import LinearSolver
from igaops.solvers.utils.argsclass import SolverArgs

logger = logging.getLogger(__name__)
array_like = Union[np.ndarray, sp.csr_array, LinearOperator]


class Template(ABC):
    "Template class for Lagrange solvers"

    def __init__(
        self,
        constraint_matrix: array_like,
        constraint_vector: np.ndarray,
        tolerance: float,
        maxiters: int,
        verbose: bool = True,
        dual_constraint_matrix: Optional[array_like] = None,
    ):
        self._config = SolverArgs(tolerance=tolerance, maxiters=maxiters)
        self._constraint_matrix = constraint_matrix
        self._constraint_vector = constraint_vector
        self._dual_matrix = constraint_matrix
        if dual_constraint_matrix is not None:
            self._dual_matrix = dual_constraint_matrix
        self._linear_solver: Optional[LinearSolver] = None
        self._lagrange_penalty: Optional[float] = None
        self._kernel: Optional[array_like] = None
        self._verbose = verbose

        # Private variable
        self._keep_warning: bool = True

    @property
    def config(self) -> SolverArgs:
        return self._config

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def linear_solver(self) -> LinearSolver:
        if not isinstance(self._linear_solver, LinearSolver):
            linear_solver = LinearSolver(
                tolerance=self.config.tolerance,
                maxiters=self.config.maxiters,
                linear_type="gmres",
                verbose=self.verbose,
            )
            self._linear_solver = linear_solver
        return self._linear_solver

    @property
    def constraint_matrix(self) -> Any:
        return self._constraint_matrix

    @property
    def dual_matrix(self) -> Any:
        return self._dual_matrix

    @property
    def constraint_vector(self) -> np.ndarray:
        return self._constraint_vector

    @property
    def lagrange_penalty(self) -> float:
        if self._lagrange_penalty is None:
            if self._keep_warning:
                logger.warning(
                    "Lagrange penalty is not set. It returns 0.0 by default."
                )
            return 0.0
        return self._lagrange_penalty

    def _get_solution(
        self, Afun: Callable, bvec: np.ndarray, Pfun: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Solves a linear system using the provided matrix-free operator.
        """
        self.linear_solver.update(
            tolerance=self.config.tolerance, maxiters=self.config.maxiters
        )
        return self.linear_solver.solve(Afun, bvec, Pfun).solution

    def _compute_dual_residual(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the residual due to constraints: g - C @ x.
        """
        g = self.constraint_vector
        return g - self.constraint_matrix @ x

    def update(
        self,
        tolerance: Optional[float] = None,
        maxiters: Optional[float] = None,
        lagrange_penalty: Optional[float] = None,
    ):
        """
        Update solver parameters and optionally Lagrange penalty.

        Parameters
        ----------
        tolerance : float, optional
            New convergence tolerance
        maxiters : int, optional
            New maximum number of iterations
        lagrange_penalty : Optional[float]
            New Lagrange penalty to be used in the augmented Lagrange multiplier method.
            If None, the penalty will not be updated.
        """
        kwargs = self.config.__dict__.copy()
        if tolerance is not None:
            kwargs["tolerance"] = tolerance
        if maxiters is not None:
            kwargs["maxiters"] = maxiters
        self._config = SolverArgs(**kwargs)

        if self.lagrange_type != "augmented":
            # If the method is not augmented, we can skip the update
            return

        if not np.isscalar(lagrange_penalty):
            # If the penalty is not a scalar, we can skip the update
            return

        # Ensure the penalty is positive
        penalty = float(np.abs(lagrange_penalty))

        if penalty <= 0.0:
            logger.warning("Lagrange penalty should be positive.")
            return

        # To avoid unnecessary update, we only update if the penalty differs significantly from the current one
        self._keep_warning = False
        rel_diff = abs(self.lagrange_penalty - penalty) / (
            abs(self.lagrange_penalty) + Constants.TINY
        )
        self._keep_warning = True

        # Update if it is not defined or the relative difference is greater than 5%
        if self._lagrange_penalty is None or rel_diff > 0.05:
            logger.info("Lagrange penalty will be updated")
            self._lagrange_penalty = penalty

    @property
    @abstractmethod
    def lagrange_type(self) -> Literal["augmented", "standard"]:
        raise NotImplementedError("To implement in children")

    @abstractmethod
    def compute_increment_lag(
        self,
        apply_T: Callable,
        apply_P: Optional[Callable],
        lagrange_res: np.ndarray,
        current_sol: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the increment according to the Lagrange method used.

        Parameters
        ----------
        apply_T : callable
            The tangent matrix (in the sense of structural mechanics).
        apply_P : callable, optional
            A good preconditioner for matrix T.
        lagrange_res : ndarray
            The residual that includes constraints.
        current_sol : ndarray, optional
            Current solution vector, required for ALM method.

        Returns
        --------
        Tuple[ndarray,ndarray]
            The increment for updating.
        """
        raise NotImplementedError("To implement in children")

    @abstractmethod
    def compute_residual_lag(
        self,
        current_res: np.ndarray,
        lagrange_multiplier: np.ndarray,
        current_sol: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute the residual incorporating the constraints according to the Lagrange method used.

        Parameters
        ----------
        current_res : ndarray
            The residual of nonlinear problem, it does not include constraints.
        lagrange_multiplier : ndarray
            Current Lagrange multiplier vector.
        current_sol : ndarray, optional
            Current solution vector, required for ALM method.

        Returns
        -------
        ndarray
            The residual updated.
        """
        raise NotImplementedError("To implement in children")
