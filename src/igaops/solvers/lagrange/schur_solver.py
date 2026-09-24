from typing import Literal
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

    # TODO: find a good preconditioner !
    # Random lower-rank preconditioner is promissing but requieres more study in our applications.
    # NOTE: Eventually, we can include a GCRODR solver to reuse Ritz eigenvectors and save some time.
    # NOTE: the convergence of the problem depends on the accuracy of the linear solver:
    # with a loose tolerance it may converge slowly

    def __init__(
        self,
        A: LinearOperator,
        mu: float,
        solver_type: Literal["lu", "polynomial", "randomized"],
    ):
        self._Amat = A
        self._mu = mu
        validate_entry(solver_type, ["lu", "polynomial", "randomized"])

        # Vanilla preconditioner
        self._preconditioner = LinearOperator(
            dtype=float, shape=A.shape, matvec=lambda x: x
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
        eye = np.eye(self._Amat.shape[0])

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
            dtype=float, shape=self._Amat.shape, matvec=lambda x: self.matvec(x)
        )
        self._preconditioner = GMRESPolynomialPreconditioner(A=matvec, degree=10)
        logger.info(f"GMRES polynomial in {time() - start:.2e} seconds.")

    def _build_randomized(self):
        random = RandomPreconditioner(
            S=self._Amat,
            mu=self._mu,
            random_type="general",
            rank=-1,
            small_threshold=0,
        )
        random.build()
        self._preconditioner = LinearOperator(
            dtype=float, shape=self._Amat.shape, matvec=lambda x: random.apply(x)
        )

    def matvec(self, x: np.ndarray):
        return self._mu * x + np.asarray(self._Amat @ x)

    def preconditioner(self, x: np.ndarray):
        return self._preconditioner @ x
