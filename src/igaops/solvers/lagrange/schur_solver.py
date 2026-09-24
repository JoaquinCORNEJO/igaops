from typing import Literal, Any, Optional, Callable
from time import time
import logging

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor, lu_solve
import numpy as np

from igaops.common import validate_entry
from igaops.solvers.linear_solver import LinearSolver
from igaops.solvers.sketch_preconditioner import RandomPreconditioner
from igaops.solvers.polynomial.preconditioner import GMRESPolynomialPreconditioner

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
    A : callable
        Matrix-free function to apply A.
    B : array_like
        Matrix B.
    C : array_like
        Matrix C.
    AT : callable
        Matrix-free function to apply A transpose. Default to None.
        It is necessary when using ``randomized`` solver type.
    mu : float
        Scalar factor to define matrix D.
    solver_type : {"lu", "polynomial", "randomized"}
        Type of solver: dense (LU decomposition) or iterative (polynomial or randomized).
    CAB_to_dense : callable, optional
        If using a dense solver, one can define how to compute the matrix C @ A (B.T).
        Otherwise, class uses A, B, C as already set. Defaults to None.
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
        A: Callable,
        B: Any,
        C: Any,
        AT: Optional[Callable],
        mu: float,
        solver_type: Literal["lu", "polynomial", "randomized"],
        CAB_to_dense: Optional[Callable] = None,
    ):
        validate_entry(solver_type, ["lu", "polynomial", "randomized"])

        # Save entry data
        self._Amat = A
        self._Bmat = B
        self._Cmat = C
        self._ATmat = AT
        self._mu = mu
        self._CAB_to_dense = CAB_to_dense

        # Define matvec linear operator
        n = B.shape[0]
        args = dict(matvec=lambda x: C @ A(B.T @ x))
        if callable(AT):
            args.update(rmatvec=lambda x: B @ AT(C.T @ x))
        self._apply_CAB = LinearOperator(dtype=float, shape=(n, n), **args)
        self._shape = (n, n)

        # Vanilla preconditioner
        self._preconditioner = LinearOperator(
            dtype=float, shape=(n, n), matvec=lambda x: x
        )

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
        logger.info(f"Schur matrix's size: {(n, n)}")

    def _build_lu(self):
        # Compute matrix
        start = time()

        eye = np.eye(self._shape[0])
        if callable(self._CAB_to_dense):
            matrix = self._CAB_to_dense(self._Bmat, self._Cmat, self._Amat)
            matrix += self._mu * eye
        else:
            matrix = self.matvec(eye)
        logger.info(f"Assemble matrix in {time() - start:.2e} seconds.")

        # Symmetrise against floating-point noise
        start = time()
        lu = lu_factor(matrix)
        logger.info(f"LU factorization in {time() - start:.2e} seconds.")

        self._preconditioner = LinearOperator(
            dtype=float, shape=matrix.shape, matvec=lambda x: lu_solve(lu, x)
        )

    def _build_polynomial(self):
        start = time()
        matvec = LinearOperator(
            dtype=float, shape=self._shape, matvec=lambda x: self.matvec(x)
        )
        self._preconditioner = GMRESPolynomialPreconditioner(A=matvec, degree=10)
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
        self._preconditioner = LinearOperator(
            dtype=float, shape=self._shape, matvec=lambda x: random.apply(x)
        )

    def matvec(self, x: np.ndarray):
        return self._mu * x + np.asarray(self._apply_CAB @ x)

    def preconditioner(self, x: np.ndarray):
        return self._preconditioner @ x

    def solve(self, x: np.ndarray):
        output = self.tight_solver.solve(self.matvec, x, self.preconditioner)
        return output.solution
