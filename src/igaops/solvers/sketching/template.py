from abc import ABC, abstractmethod
from typing import Optional, Any
from time import time
import logging

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor, lu_solve
import numpy as np

from igaops.common import Constants
from .operations import (
    gaussian_test_matrix,
    adaptive_range_finder,
    power_iteration,
)

logger = logging.getLogger(__name__)


class Template(ABC):
    "Reference class for random preconditioner"

    def __init__(
        self,
        S: LinearOperator,
        mu: float,
        rank: int = 24,
        small_threshold: int = 576,
        random_state: int = 0,
    ):
        self.S = S
        self.mu = float(mu)
        self.rank = int(rank)
        self.small_threshold = int(small_threshold)
        self.opshape = S.shape

        self._built = False
        self._dense = False
        self._rng = np.random.default_rng(random_state)
        self._oversampling = 10

        # Dense mode
        self._LU = np.array([])

        # Iterative mode
        self._LAMBDA = np.array([])
        self._U = np.array([])
        self._V = np.array([])
        self._pseudo_inverse: Optional[Any] = None

        # NOTE: The following values seems to be problem-dependent
        # TODO: carry out heuristic study in IGA
        self._low_rank_tolerance: float = 1e-2
        self._power_iters: int = 1

    @property
    def is_built(self):
        return self._built

    @property
    def is_dense(self):
        return self._dense

    @property
    def auto_range_finder(self):
        return False if self.rank > 0 else True

    @property
    def should_build_dense(self):
        return self.opshape[0] < self.small_threshold or self.rank >= self.opshape[0]

    @property
    def choosen_method(self):
        return "Adaptive randomized" if self.auto_range_finder else "Full randomized"

    @property
    def Lambda(self):
        return self._LAMBDA

    @property
    def low_rank_ratio(self) -> float:
        if self.low_rank_size == 0:
            return 1.0
        valmax = self.Lambda[0]
        valmin = self.Lambda[-1]
        return valmin / (valmax + Constants.SAFEGUARD)

    @property
    def low_rank_size(self) -> int:
        return len(self.Lambda)

    @property
    def low_rank_tolerance(self) -> float:
        if self._low_rank_tolerance is None:
            raise RuntimeError("Value not defined")
        return self._low_rank_tolerance

    @property
    def power_iters(self) -> int:
        if self._power_iters is None:
            raise RuntimeError("Value not defined")
        return self._power_iters

    # ------------------------------------------------------------------
    # Dense build
    # ------------------------------------------------------------------
    def _build_dense(self):
        # Define eye
        eye = np.eye(self.opshape[0])

        # Compute matrix
        start = time()
        SS = self.S @ eye
        logger.info(f"Assemble matrix in {time() - start:.2e} seconds.")

        # Symmetrise against floating-point noise
        SS += self.mu * eye
        start = time()
        LU = lu_factor(SS)
        logger.info(f"LU factorization in {time() - start:.2e} seconds.")

        self._LU = LU
        self._dense = True
        self._built = True

    # ------------------------------------------------------------------
    # Random low-rank build
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_low_rank(self):
        raise NotImplementedError("To implement in children")

    # ------------------------------------------------------------------
    # Public build
    # ------------------------------------------------------------------
    def _find_low_rank_Q(self):
        ovsmp = self._oversampling
        if self.auto_range_finder:
            Q = adaptive_range_finder(
                self.S,
                n=self.opshape[0],
                sample_size=ovsmp,
                relative_tolerance=self.low_rank_tolerance,
                rng=self._rng,
                power_iters=self.power_iters,
            )
            r = Q.shape[1]
        else:
            r = min(self.rank, *self.opshape)
            k = min(r + ovsmp, *self.opshape)
            Omega = gaussian_test_matrix(
                n=self.opshape[0],
                sample_size=k,
                rng=self._rng,
            )
            Y = power_iteration(
                self.S,
                Omega,
                power_iters=self.power_iters,
            )
            Q, _ = np.linalg.qr(Y, mode="reduced")
        return np.asarray(Q), r

    def build(self):
        """Build the preconditioner."""
        start = time()
        if self.should_build_dense:
            self._build_dense()
        else:
            self._build_low_rank()
        message = repr(self) + f" in {time() - start:.2e} seconds."
        logger.info(message)

    # ------------------------------------------------------------------
    # Apply  P^{-1}
    # ------------------------------------------------------------------
    @abstractmethod
    def _solve_low_rank(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("To implement in children")

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the preconditioner to an array.

        Parameters
        ----------
        x : ndarray
            Input array to whom the preconditioner is applied.

        Returns
        -------
        ndarray
            The preconditioner apply to the input array.
        """
        if not self.is_built:
            raise RuntimeError("Call build() before applying the preconditioner.")

        if self.is_dense:
            return lu_solve(self._LU, x)

        return self._solve_low_rank(x)

    def _get_info(self, name):

        if not self.is_built:
            return f"{name}(not built)"

        if self.is_dense:
            return f"{name}(dense, shape={self.opshape}, mu={self.mu:.2e})"

        tolerance = (
            f"{self.low_rank_tolerance:.2e}"
            if self.auto_range_finder
            else "not concerned"
        )
        return (
            f"{name}("
            f"sketch {self.choosen_method}, "
            f"shape={self.opshape}, "
            f"mu={self.mu:.2e}, "
            f"rank={self.low_rank_size}, "
            f"spectral_ratio={self.low_rank_ratio:.2e}, "
            f"tolerance={tolerance})"
        )
