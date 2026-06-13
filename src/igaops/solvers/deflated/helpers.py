from typing import Callable

from scipy import linalg as sclin
import numpy as np

from igaops.common import Constants

# ---------------------------------------------------------------------------
# gmres1  –  plain preconditioned GMRES (priming cycle)
# ---------------------------------------------------------------------------


def gmres1(
    A_apply: Callable,
    x: np.ndarray,
    r: np.ndarray,
    m: int,
    M_apply: Callable,
    tol: float,
):
    """
    Run up to m steps of preconditioned GMRES starting from (x, r).
    Generates  A V[:, :m] = V[:, :m+1] H.
    """
    dtype = np.complex128
    x = np.asarray(x, dtype=dtype)
    r = np.asarray(r, dtype=dtype)

    n = len(r)
    V = np.zeros((m + 1, n), dtype=dtype)
    H = np.zeros((m + 1, m), dtype=dtype)
    resvec = np.zeros(m)

    k = 1
    beta = np.linalg.norm(r)
    y = np.zeros(1, dtype=dtype)
    res = np.array([beta])

    if beta <= Constants.SAFEGUARD:
        return x, r, V, H, k, resvec[:k]

    V[0] = r / beta
    for k in range(1, m + 1):
        w = M_apply(A_apply(V[k - 1]))

        for j in range(k):
            hij = V[j].conj().dot(w)
            H[j, k - 1] += hij
            w -= hij * V[j]

        # reorthogonalization pass
        for j in range(k):
            hij = V[j].conj().dot(w)
            H[j, k - 1] += hij
            w -= hij * V[j]

        norm_w = np.linalg.norm(w)
        H[k, k - 1] = norm_w
        if norm_w > 0.0:
            V[k] = w / norm_w

        mat = H[: k + 1, :k]
        rhs = np.zeros(k + 1, dtype=dtype)
        rhs[0] = beta
        y = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        res = rhs - mat @ y
        resvec[k - 1] = np.linalg.norm(res)

        if resvec[k - 1] < tol:
            break

    x += V[:k].T @ y
    r = V[: k + 1].T @ res
    return x, r, V, H, k, resvec[:k]


# ---------------------------------------------------------------------------
# gmres2  –  deflated Arnoldi for the main GCRO-DR cycle
# ---------------------------------------------------------------------------


def gmres2(
    A_apply: Callable,
    r: np.ndarray,
    m: int,
    M_apply: Callable,
    C: np.ndarray,
    tol: float,
):
    """
    Run up to m deflated Arnoldi steps.
    Generates  (I - C Cᵀ) M⁻¹ A V[:, :m] = V[:, :m+1] H.

    B[:, j] = Cᵀ (M⁻¹ A v_j)  is the cross-term coupling C to V.
    """
    dtype = np.complex128
    r = np.asarray(r, dtype=dtype)
    C = np.asarray(C, dtype=dtype)

    n = len(r)
    V = np.zeros((m + 1, n), dtype=dtype)
    H = np.zeros((m + 1, m), dtype=dtype)
    B = np.zeros((C.shape[1], m), dtype=dtype)
    resvec = np.zeros(m)

    k = 1
    beta = np.linalg.norm(r)
    y = np.zeros(1)
    res = np.array([beta])

    if beta <= Constants.SAFEGUARD:
        return V, H, B, k, resvec[:k]

    V[0] = r / beta
    for k in range(1, m + 1):
        w = M_apply(A_apply(V[k - 1]))

        B[:, k - 1] = C.conj().T @ w  # record C-component before deflation
        w = w - C @ B[:, k - 1]  # deflate

        for j in range(k):
            hij = V[j].conj().dot(w)
            H[j, k - 1] += hij
            w -= hij * V[j]

        # reorthogonalization pass
        for j in range(k):
            hij = V[j].conj().dot(w)
            H[j, k - 1] += hij
            w -= hij * V[j]

        norm_w = np.linalg.norm(w)
        H[k, k - 1] = norm_w
        if norm_w > 0.0:
            V[k] = w / norm_w

        mat = H[: k + 1, :k]
        rhs = np.zeros(k + 1, dtype=dtype)
        rhs[0] = beta
        y = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        res = rhs - mat @ y
        resvec[k - 1] = np.linalg.norm(res)
        if resvec[k - 1] < tol:
            break

    return V, H, B, k, resvec[:k]


# ---------------------------------------------------------------------------
# getHarmVecs1  –  harmonic Ritz extraction after a plain GMRES run
# ---------------------------------------------------------------------------


def getHarmVecs1(m: int, k: int, H: np.ndarray):
    """
    Extract k harmonic Ritz vectors from the (m+1)xm Hessenberg H.

    Harmonic Ritz matrix:
        G = Hm  +  h_{m+1,m}²  (Hm^{-T} eₘ) eₘᵀ

    Returns the k eigenvectors of G with *smallest* |eigenvalue|.

    Parameters
    ----------
    m : int
        Number of GMRES steps  (H is (m+1)xm)
    k : int
        Number of vectors to return
    H : ndarray
        (m+1, m) upper Hessenberg

    Returns
    -------
    ndarray
        Matrix of size (m, k)
    """
    Hm = H[:m, :m]  # m × m  (square upper Hessenberg)

    em = np.zeros(m, dtype=np.complex128)
    em[-1] = 1.0
    inv_Hm_H_em = np.linalg.solve(Hm.conj().T, em)  # Hm^{-H} eₘ

    h = np.abs(H[m, m - 1]) ** 2
    harmRitzMat = Hm + h * np.outer(inv_Hm_H_em, em.conj())

    eigvals, eigvecs = sclin.eig(harmRitzMat)

    # remove infinities / NaNs
    good = np.isfinite(eigvals)
    eigvals = eigvals[good]
    eigvecs = eigvecs[:, good]

    # choose smallest harmonic Ritz values
    idx = np.argsort(np.abs(eigvals))

    return eigvecs[:, idx[:k]]


# ---------------------------------------------------------------------------
# getHarmVecs2  –  harmonic Ritz extraction inside the main GCRO-DR loop
# ---------------------------------------------------------------------------


def getHarmVecs2(
    m: int,
    k: int,
    G: np.ndarray,
    V: np.ndarray,
    U: np.ndarray,
    C: np.ndarray,
):
    """
    Harmonic Ritz extraction for GCRO-DR (Parks et al.)

    Parameters
    ----------
    m : int
        Total augmented subspace dimension (= k + p)
    k : int
        Number of vectors to recycle
    G : ndarray
        (m+1, m) augmented Hessenberg matrix
    V : ndarray
        (p+1, n) Krylov basis including extra Arnoldi vector
    U : ndarray
        (n, k) recycled basis
    C : ndarray
        (n, k) orthonormal image basis, C = M^{-1} A U

    Returns
    -------
    ndarray
        Matrix P of size (m, k). Contains the coefficient vectors
        defining next recycle space.
    """

    p = m - k

    # Search basis
    Vtilde = np.hstack([U, V[:p].T])

    # Image basis
    Wtilde = np.hstack([C, V[: p + 1].T])

    # Exact overlap matrix
    Phi = Wtilde.conj().T @ Vtilde

    # Harmonic Ritz generalized EVP:
    #
    #  (H2^* H2) y = theta (H2^* Phi) y
    #
    A = G.conj().T @ G
    B = G.conj().T @ Phi

    eigvals, eigvecs = sclin.eig(A, B)

    # remove infinities / NaNs
    good = np.isfinite(eigvals)
    eigvals = eigvals[good]
    eigvecs = eigvecs[:, good]

    # choose smallest harmonic Ritz values
    idx = np.argsort(np.abs(eigvals))

    return eigvecs[:, idx[:k]]
