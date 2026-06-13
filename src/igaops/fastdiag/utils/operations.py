from typing import Optional
import numpy as np


def compute_special_schur(
    H: np.ndarray,
    bfac: np.ndarray,
) -> np.ndarray:
    """
    Computes the Schur complement of the special arrow-head structure
    use in space-time preconditioning.

    Parameters
    ----------
    H : ndarray
        List of diagonals H_i of shape (m,), for i = 1, ..., k+1. Its shape is (k+1, m).
    bfac : ndarray
        List of scalars coefficients b_i. Its shape is (k,)

    Returns
    -------
    ndarray
        Schur complement.
    """

    H_top = H[:-1, :]  # shape k, m
    H_last = H[-1, :]  # shape m

    # Schur complement
    schur = H_last + np.sum((np.abs(bfac) ** 2)[:, np.newaxis] / H_top, axis=0)

    return schur


def solve_special_arrowhead(
    H: np.ndarray,
    bfac: np.ndarray,
    rhs: np.ndarray,
    schur: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solves arrow-head matrix used in space-time preconditioning for multiples right-hand-sides.

    Parameters
    ----------
    H : ndarray
        List of diagonals H_i of shape (m,), for i = 1, ..., k+1. Its shape is (k+1, m).
    bfac : ndarray
        List of scalars coefficients b_i. Its shape is (k,)
    rhs : ndarray
        List of righ-hand-sides stacked in columns, i.e., its shape is ((k+1)*m, r).
    schur : ndarray, optional
        The precomputed Schur complement. If it is not defined, it is evaluated on the fly.

    Returns
    -------
    ndarray
        The solution for the different rhs.
    """
    if not 0 < rhs.ndim < 3:
        raise ValueError("rhs expected to be 1 and 2 array.")

    ndim_is_one = rhs.ndim == 1

    H_top = H[:-1, :]  # shape k, m
    r = 1 if ndim_is_one else rhs.shape[1]

    rhs = np.reshape(rhs, (*H.shape, r))  # shape k+1, m, r
    rhs_top = rhs[:-1, ...]  # shape k, m, r
    rhs_last = rhs[-1, ...]  # shape m, r

    # y_i = H_i^{-1} rhs_i
    y = rhs_top / H_top[..., np.newaxis]  # shape k, m, r

    # Schur complement
    if schur is None:
        schur = compute_special_schur(H, bfac)  # shape m

    # RHS of last block: shape m, r
    rhs_s = rhs_last + np.einsum("i,i...->...", np.conj(bfac), y, optimize=True)

    x_last = rhs_s / np.asarray(schur)[:, np.newaxis]  # shape m, r

    # back substitution
    multp = bfac[:, np.newaxis] / H_top  # shape k, m
    x_top = y - multp[..., np.newaxis] * x_last[np.newaxis, ...]  # shape k, m, r

    x = np.concatenate([x_top, x_last[np.newaxis, ...]], axis=0)

    if ndim_is_one:
        return np.ravel(x)

    return np.reshape(x, (-1, r))  # shape (k+1)*m, r
