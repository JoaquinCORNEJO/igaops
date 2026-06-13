from typing import Optional, Tuple

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import qr
import numpy as np

from igaops.common import Constants


# ------------------------------------------------------------------------------
# Common methods
# ------------------------------------------------------------------------------
def power_iteration(
    A: LinearOperator,
    Omega: np.ndarray,
    power_iters: int = 1,
) -> np.ndarray:
    """
    Power iteration algorithm for matrix A using vectors Omega.

    It evaluates (A A^T)^p A @ Omega.

    Parameters
    ----------
    A : LinearOperator
        Matrix to apply power iteration.
    Omega : ndarray
        Probing vectors.
    power_iters : int, optional
        Parameter p in (A A^T)^p A @ Omega. Defaults to 1.

    Raises
    ------
    TypeError
        - If A is not callable.
        - If ``power_iters`` is greater than 0 and AT is not callable.
    """
    if not isinstance(A, LinearOperator):
        raise TypeError("A should be callable")

    Y = A @ Omega
    for _ in range(power_iters):
        Q = qr(Y, mode="economic")[0]
        Ytilde = A.T @ Q
        Qtilde = qr(Ytilde, mode="economic")[0]
        Y = A @ Qtilde
    return np.asarray(Y)


def adaptive_range_finder(
    A: LinearOperator,
    n: int,
    sample_size,
    relative_tolerance: float = 1e-2,
    max_rank: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    power_iters: int = 1,
) -> np.ndarray:
    """
    Computes the low-rank space Q using an adaptive algorithm.

    Parameters
    ----------
    A : LinearOperator
        Matrix to apply adaptive range finder.
    n : int
        Column dimension of A.
    sample_size : int
        Initial value of number of columns of Q matrix.
    relative_tolerance : float
        Tolerance to stop algorithm. Defaults to 1e-2.
    max_rank : int
        Maximal number of columns of Q matrix.
    rng : Optional[np.random.Generator]
        Random generator.
    power_iters : int
        Parameter for power iteration. Defaults to 1.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def eval_error(x):
        return np.max(np.linalg.norm(x, axis=0))

    Omega_pool = rng.standard_normal((n, sample_size))
    Y_pool = power_iteration(A, Omega_pool, power_iters=power_iters)

    m = Y_pool.shape[0]

    if max_rank is None:
        max_rank = min(m, n)

    max_rank = int(min(max_rank, m, n))

    # Set absolute threshold from the initial residual — no external norm needed
    initial_max_norm = eval_error(Y_pool)
    thresh = relative_tolerance * initial_max_norm / (10 * np.sqrt(2 / np.pi))

    Q = np.empty((m, 0))
    for _ in range(max_rank):

        if eval_error(Y_pool) <= thresh:
            break

        y_raw = Y_pool[:, 0].copy()
        y = y_raw.copy()

        if Q.shape[1]:
            y -= Q @ (Q.T @ y)

        ny = np.linalg.norm(y)
        if ny < Constants.SAFEGUARD:
            Omega_pool = Omega_pool[:, 1:]
            Y_pool = Y_pool[:, 1:]
            omega_new = rng.standard_normal(n)
            y_new = power_iteration(
                A,
                omega_new[:, None],
                power_iters=power_iters,
            ).ravel()
            if Q.shape[1]:
                y_new -= Q @ (Q.T @ y_new)
            Omega_pool = np.column_stack((Omega_pool, omega_new))
            Y_pool = np.column_stack((Y_pool, y_new))
            continue

        q = y / ny
        Q = np.column_stack((Q, q))

        # remove consumed column
        Omega_pool = Omega_pool[:, 1:]
        Y_pool = Y_pool[:, 1:]

        # downdate pool
        if Y_pool.shape[1]:
            Y_pool -= np.outer(q, q @ Y_pool)

        # replenish
        omega_new = rng.standard_normal(n)
        y_new_raw = power_iteration(
            A,
            omega_new[:, None],
            power_iters=power_iters,
        ).ravel()

        y_new = y_new_raw.copy()
        y_new -= Q @ (Q.T @ y_new)

        Omega_pool = np.column_stack((Omega_pool, omega_new))
        Y_pool = np.column_stack((Y_pool, y_new))

    Q = qr(Q, mode="economic")[0]
    return np.asarray(Q)


def gaussian_test_matrix(
    n: int,
    sample_size: int,
    orthonormal: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a Gaussian test matrix

    Parameters
    ----------
    n : int
        Number of rows of Gaussian matrix.
    sample_size : int
        Number of columns of Gaussian matrix.
    orthonormal : bool
        If ``True``, it computes QR decomposition of Gaussian matrix. Defaults to ``False``.
    rng : Optional[np.random.Generator]
        Random generator.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    Omega = rng.standard_normal((n, sample_size)) / np.sqrt(n)

    if orthonormal:
        Omega = qr(Omega, mode="economic")[0]

    return np.asarray(Omega)


# ------------------------------------------------------------------------------
# Methods for SPD matrices
# ------------------------------------------------------------------------------
def eigen_decomposition(
    A: LinearOperator, Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Low-rank Eigen-decomposition from a left range basis Q.

        A ≈ Q Qᵀ A  ⟹  B = Qᵀ A = (Aᵀ Q)ᵀ

    Then B = Uhat Σ Uhatᵀ and U = Q Uhat.

    Parameters
    ----------
    A : LinearOperator
    Q : ndarray

    Notes
    -----
    It is assumed that the matrix is symetric, i.e. Aᵀ = A
    """

    AQ = A @ Q
    B = Q.T @ AQ
    B = 0.5 * (B + B.T)

    eigvals, eigvecs = np.linalg.eigh(B)

    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    Sigma = np.maximum(eigvals, 0.0)

    U = Q @ eigvecs
    return U, Sigma


# ------------------------------------------------------------------------------
# Methods for non symmetric matrices
# ------------------------------------------------------------------------------
def svd_decomposition(
    A: LinearOperator, Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Low-rank SVD from a left range basis Q.

        A ≈ Q Qᵀ A  ⟹  B = Qᵀ A = (Aᵀ Q)ᵀ

    Then B = Uhat Σ Vᵀ and  U = Q Uhat.

    Parameters
    ----------
    A : LinearOperator
    Q : ndarray
    """
    B = A.T @ Q

    Uhat, Sigma, VT = np.linalg.svd(B.T, full_matrices=False)
    U = Q @ Uhat

    return U, Sigma, VT.T
