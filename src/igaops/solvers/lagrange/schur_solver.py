from typing import Literal, Any
from time import time
import logging

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor, lu_solve
import numpy as np

from igaops.common import validate_entry
from igaops.solvers.linear_solver import LinearSolver
from igaops.solvers.sketch_preconditioner import RandomPreconditioner
from igaops.solvers.polynomial.preconditioner import GMRESPolynomialPreconditioner

from .template import preconditioner_like

logger = logging.getLogger(__name__)


class SchurSolver:
    """
    Solver for the (negative) Schur complement.

    It arises when trying to solve the following system:
        [A      B^T]
        [C      -D ]
    Here, the Schur matrix is -S = D + C A^-1 B^T.
    From now on, we only consider D = mu * I.

    Parameters
    ----------
    P : preconditioner_like
        Class of preconditioner with methods apply and apply_transpose.
        Method ``apply`` represents block A and ``apply_transpose`` the block A^T.
    B : array_like
        Block matrix B.
    C : array_like
        Block matrix C.
    mu : float
        Scalar factor to define block matrix D.
    solver_type : {"lu", "polynomial", "randomized"}
        Type of solver: dense (LU decomposition) or iterative (polynomial or randomized).
    """

    # NOTE 1: The convergence of ALM depends on the accuracy of the linear solver: with a loose tolerance it may even diverge.
    # NOTE 2: A GCRODR solver seems reasonable to reuse Ritz eigenvectors and save some time (for polinomial and randomized).
    tight_solver = LinearSolver(
        tolerance=1e-8,
        maxiters=100,
        linear_type="gmres",
        verbose=False,
    )

    def __init__(
        self,
        P: preconditioner_like,
        B: Any,
        C: Any,
        mu: float,
        solver_type: Literal["lu", "polynomial", "randomized"],
    ):
        validate_entry(solver_type, ["lu", "polynomial", "randomized"])

        # Save entry data
        self._precond = P
        self._Bmat = B
        self._Cmat = C
        self._mu = mu

        # Define matvec linear operator
        n = B.shape[0]
        self._shape = (n, n)
        self._apply_CAB = self._set_linear_operator()
        self._schur_precond = None

        # Change preconditioner according to solver_type
        if solver_type == "lu":
            # NOTE: It works pretty well but it is expensive.
            self._build_lu()
        elif solver_type == "polynomial":
            # FIXME: Polynomial decomposition needs to be revisited.
            self._build_polynomial()
        elif solver_type == "randomized":
            # NOTE: Random lower-rank preconditioner is promissing but
            # requieres more study in our applications, specially to tune the hyper-parameters.
            self._build_randomized()
        else:
            raise ValueError("Unknown method")
        logger.info(f"Schur matrix's size: {self._shape}")

    def _set_linear_operator(self):
        P = self._precond
        B = self._Bmat
        C = self._Cmat
        shape = self._shape
        linop = LinearOperator(
            dtype=float,
            shape=shape,
            matvec=lambda x: C @ P.apply(B.T @ x),
            rmatvec=lambda x: B @ P.apply_transpose(C.T @ x),
        )
        return linop

    def _build_lu(self):
        start = time()
        eye = np.eye(self._shape[0])
        matrix = self.matvec(eye)
        logger.info(f"Assemble matrix in {time() - start:.2e} seconds.")

        start = time()
        lu = lu_factor(matrix)
        logger.info(f"LU factorization in {time() - start:.2e} seconds.")

        self._schur_precond = LinearOperator(
            dtype=float, shape=matrix.shape, matvec=lambda x: lu_solve(lu, x)
        )

    def _build_polynomial(self):
        start = time()
        matvec = LinearOperator(dtype=float, shape=self._shape, matvec=self.matvec)
        self._schur_precond = GMRESPolynomialPreconditioner(A=matvec, degree=10)
        logger.info(f"GMRES polynomial in {time() - start:.2e} seconds.")

    def _build_randomized(self):
        # In what follows, a negative rank enables adaptive range finder
        # A null small threshold forces to use iterative method (since LU is already in this class)
        random = RandomPreconditioner(
            S=self._apply_CAB,
            mu=self._mu,
            random_type="general",
            rank=-1,
            small_threshold=0,
        )
        random.build()
        self._schur_precond = LinearOperator(
            dtype=float, shape=self._shape, matvec=random.apply
        )

    def matvec(self, x: np.ndarray):
        return self._mu * x + np.asarray(self._apply_CAB @ x)

    def solve(self, x: np.ndarray):
        output = self.tight_solver.solve(self.matvec, x, self._schur_precond)
        return output.solution
