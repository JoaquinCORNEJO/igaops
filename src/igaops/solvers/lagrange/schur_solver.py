from typing import Literal, Any, Optional, Callable
from time import time
import logging

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor, lu_solve
import numpy as np

from igaops.common import validate_entry
from igaops.solvers.sketch_preconditioner import RandomPreconditioner
from igaops.solvers.polynomial.preconditioner import GMRESPolynomialPreconditioner

logger = logging.getLogger(__name__)


class SchurSolver:
    """
    Solver for the (negative) Schur complement that arises from
    [A  B^T]
    [C  -D ]
    That is -S = D + C A^-1 B^T. In this class we consider D = mu * I.
    """

    # Random lower-rank preconditioner is promissing but requieres more study in our applications.
    # NOTE: Eventually, we can include a GCRODR solver to reuse Ritz eigenvectors and save some time.
    # The convergence depends on the accuracy of the linear solver:  with a loose tolerance it may even diverge.

    def __init__(
        self,
        A: Callable,
        B: Any,
        C: Any,
        AT: Optional[Callable],
        mu: float,
        solver_type: Literal["lu", "polynomial", "randomized"],
    ):
        n = B.shape[0]
        args = dict(matvec=lambda x: C @ A(B.T @ x))
        if callable(AT):
            args.update(rmatvec=lambda x: B @ AT(C.T @ x))

        self._matvec = LinearOperator(dtype=float, shape=(n, n), **args)
        self._mu = mu
        validate_entry(solver_type, ["lu", "polynomial", "randomized"])

        # Vanilla preconditioner
        self._preconditioner = LinearOperator(
            dtype=float, shape=(n, n), matvec=lambda x: x
        )

        # Change preconditioner according to solver_type
        if solver_type == "lu":
            self._build_lu()
        elif solver_type == "polynomial":
            self._build_polynomial()
        elif solver_type == "randomized":
            self._build_randomized()
        else:
            raise ValueError("Unknown method")

    def _build_lu(self):
        eye = np.eye(self._matvec.shape[0])

        # Compute matrix
        start = time()
        matrix = self.matvec(eye)
        logger.info(f"Assemble matrix in {time() - start:.2e} seconds.")

        # Symmetrise against floating-point noise
        start = time()
        lu = lu_factor(matrix)
        logger.info(f"LU factorization in {time() - start:.2e} seconds.")

        self._preconditioner = LinearOperator(
            dtype=float, shape=eye.shape, matvec=lambda x: lu_solve(lu, x)
        )

    def _build_polynomial(self):
        start = time()
        matvec = LinearOperator(
            dtype=float, shape=self._matvec.shape, matvec=lambda x: self.matvec(x)
        )
        self._preconditioner = GMRESPolynomialPreconditioner(A=matvec, degree=10)
        logger.info(f"GMRES polynomial in {time() - start:.2e} seconds.")

    def _build_randomized(self):
        random = RandomPreconditioner(
            S=self._matvec,
            mu=self._mu,
            random_type="general",
            rank=-1,
            small_threshold=0,
        )
        random.build()
        self._preconditioner = LinearOperator(
            dtype=float, shape=self._matvec.shape, matvec=lambda x: random.apply(x)
        )

    def matvec(self, x: np.ndarray):
        return self._mu * x + np.asarray(self._matvec @ x)

    def preconditioner(self, x: np.ndarray):
        return self._preconditioner @ x
