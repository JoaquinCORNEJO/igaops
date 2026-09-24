from typing import Union, Callable, Optional, Tuple
from time import time
import logging

from scipy.sparse.linalg import LinearOperator, lsqr
from scipy import linalg as sclin
from scipy import sparse as sp
import numpy as np

from igaops.common import Constants
from .template import Template

logger = logging.getLogger(__name__)


class StandardLagrange(Template):
    """
    Class of Standard Lagragian method using null space.
    """

    @property
    def is_penalty_necessary(self):
        return False

    @property
    def kernel(self):
        if self._kernel is None:
            n = self.constraint_matrix.shape[1]
            constraint_matrix = self.constraint_matrix @ np.eye(n)
            self._kernel = Helpers.compute_kernel(np.asarray(constraint_matrix))
        return self._kernel

    def compute_increment_lag(
        self,
        apply_T: Callable,
        apply_P: Optional[Callable],
        lagrange_res: np.ndarray,
        current_sol: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        residual = lagrange_res.copy()
        solution = current_sol.copy()

        def Zdot(x):
            y = self.kernel @ x
            return y

        def Zdot_T(x):
            return self.kernel.T @ x

        res_lam = self._compute_dual_residual(solution)
        delta_up = Helpers.lstsq(self.constraint_matrix, res_lam, is_transpose=False)

        def red_matvec(x_red: np.ndarray) -> np.ndarray:
            "Computes M x_red where M = (Z^T T Z)"
            x = Zdot(x_red)
            y = apply_T(x)
            return Zdot_T(y)

        def red_preconditioner(rx_red: np.ndarray) -> np.ndarray:
            "Apply preconditioner for the standard method: Z^T P Z"
            if not callable(apply_P):
                return rx_red
            # NOTE: Z @ xred is equivalent to solve Z.T y = xred
            # since Z is orthogonal, ie, Z @ Z.T = Identity
            y = Zdot(rx_red)
            w = apply_P(y)
            # NOTE: Z.T @ w is equivalent to solve Z out = w
            # since Z is orthogonal, ie, Z.T @ Z = Identity
            return Zdot_T(w)

        rhs = residual - apply_T(delta_up)
        rhs_red = Zdot_T(rhs)

        y = self._get_solution(red_matvec, rhs_red, red_preconditioner)
        delta_ug = Zdot(y)
        delta_u = delta_up + delta_ug

        Tdug = apply_T(delta_ug)
        delta_mult = Helpers.lstsq(
            self.constraint_matrix, rhs - Tdug, is_transpose=True
        )

        return delta_u, delta_mult

    def compute_residual_lag(
        self,
        current_res: np.ndarray,
        lagrange_multiplier: np.ndarray,
        current_sol: Optional[np.ndarray],
    ):
        # Current residual =  f - A @ U
        updated_res = current_res.copy()

        # For standard residual f - A @ U - C^T @ lambda
        # NOTE: This statement is true because we will later multiply Z.T @ res
        # and by definition of the null space Z.T @ C.T = (C @ Z).T should be zero
        updated_res -= self.constraint_matrix.T @ lagrange_multiplier

        return updated_res


class Helpers:

    @staticmethod
    def compute_kernel(C: Union[np.ndarray, sp.csr_array]):
        """
        Computes the nullspace (kernel) of the constraint matrix.

        Parameters
        ----------
        C : array_like
            Constraint matrix.

        Returns
        -------
        array_like
            Nullspace basis.
        """
        # TODO: find kernel directly for sparse matrices.
        # There is a functionality in scipy but there are a lot of issues
        # when I was developing this class
        if not (isinstance(C, np.ndarray) or sp.issparse(C)):
            raise TypeError("Matrix should be ndarray or sparse.")
        start = time()
        mat = C.copy() if isinstance(C, np.ndarray) else C.toarray()
        ker = sclin.null_space(mat)
        logger.info(
            f"Null space of constraint matrix computed in {time() - start:.2e} seconds"
        )
        return Helpers.maybe_convert_sparse(ker)

    @staticmethod
    def maybe_convert_sparse(
        Z: np.ndarray, density_threshold: float = 0.10, memory_saving: float = 0.8
    ) -> Union[np.ndarray, sp.csr_array]:
        """
        Converts a dense matrix to sparse if it is memory efficient.

        Parameters
        ----------
        Z : ndarray
            Matrix to check.
        density_threshold : float
            Density threshold for conversion.
        memory_saving : float
            Minimum memory saving ratio.

        Returns
        -------
        array_like
            Converted matrix.
        """
        if sp.issparse(Z):
            return Z
        Z_csr = sp.csr_array(Z)
        Z_csr.eliminate_zeros()
        sparse_bytes = Z_csr.data.nbytes + Z_csr.indices.nbytes + Z_csr.indptr.nbytes
        dense_bytes = Z.nbytes
        density = Z_csr.nnz / Z.size
        if density < density_threshold or sparse_bytes < memory_saving * dense_bytes:
            logger.debug(
                f"Converting to sparse: density={100*density:.2f}%, "
                f"dense={dense_bytes/1e6:.2f} MB, sparse={sparse_bytes/1e6:.2f} MB"
            )
            return Z_csr
        logger.debug(
            f"Keeping dense: density={100*density:.2f}%, "
            f"dense={dense_bytes/1e6:.2f} MB, sparse={sparse_bytes/1e6:.2f} MB"
        )
        return Z

    @staticmethod
    def lstsq(
        M: Union[np.ndarray, sp.csr_array, LinearOperator],
        rhs: np.ndarray,
        is_transpose=False,
    ) -> np.ndarray:
        """
        Solves a least squares problem.

        Parameters
        ----------
        M : array_like
            Matrix.
        rhs : ndarray
            Right-hand side vector.
        is_transpose : bool
            If True, uses the transpose of M i.e. M^T.

        Returns
        -------
        ndarray
            Solution vector.
        """
        tol = Constants.SAFEGUARD
        mat = M.T if is_transpose else M
        return lsqr(mat, rhs, atol=tol, btol=tol)[0]
