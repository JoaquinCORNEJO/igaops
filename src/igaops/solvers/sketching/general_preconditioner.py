from scipy.linalg import lu_factor, lu_solve
import numpy as np

from .template import Template
from .operations import svd_decomposition


class RandGeneralPreconditioner(Template):
    "Random preconditioner for general matrices"

    @property
    def pseudo_inverse(self):
        if self._pseudo_inverse is None:
            U, V = self._U, self._V
            matrix = self.mu * np.diag(1.0 / self.Lambda) + (V.T @ U)
            self._pseudo_inverse = lu_factor(matrix)
        return self._pseudo_inverse

    # ------------------------------------------------------------------
    # Random low-rank build
    # ------------------------------------------------------------------
    def _build_low_rank(self):
        Q, r = self._find_low_rank_Q()
        U, Sigma, V = svd_decomposition(self.S, Q)
        self._U = U[:, :r]
        self._LAMBDA = Sigma[:r]
        self._V = V[:, :r]
        self._dense = False
        self._built = True

    # ------------------------------------------------------------------
    # Apply  P^{-1}
    # ------------------------------------------------------------------
    def _solve_low_rank(self, x: np.ndarray) -> np.ndarray:
        """
        Apply P⁻¹ to vector x. It implements Woodbury formula adapted to the Schur complement
        M = μI + S where S ≈ U Λ Vᵀ:

            P⁻¹ = (mu*I + U Λ V.T)^-1 = 1/mu*(I - U R^-1 V^T)

        where R is defined by R = mu * 1 / diag(Λ) + V^T. The shape of matrix R is rxr,
        where r is the rank approximation of S.

        Quality degrades when μ >> λ_max(S) (identity dominates, sketch irrelevant)
        or when rank is too small to capture the spectrum of S.
        """
        mu = self.mu
        U, V = self._U, self._V
        inv_mu_x = x / mu
        coeff = lu_solve(self.pseudo_inverse, V.T @ inv_mu_x)
        return inv_mu_x - (U @ coeff)

    def __repr__(self):
        cls = type(self).__name__
        return self._get_info(cls)
