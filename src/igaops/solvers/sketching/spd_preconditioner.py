import numpy as np

from .template import Template
from .operations import eigen_decomposition


class RandSPDPreconditioner(Template):
    "Random preconditioner for Symetric Positive Definite (SPD) matrices"

    @property
    def pseudo_inverse(self):
        if self._pseudo_inverse is None:
            mu = self.mu
            lam = self.Lambda[-1]
            invdiag = (lam + mu) / (self.Lambda + mu) - 1
            self._pseudo_inverse = invdiag
        return self._pseudo_inverse

    # ------------------------------------------------------------------
    # Random low-rank build
    # ------------------------------------------------------------------
    def _build_low_rank(self):
        Q, r = self._find_low_rank_Q()
        U, Lambda = eigen_decomposition(self.S, Q)
        self._U = U[:, :r]
        self._LAMBDA = Lambda[:r]
        self._dense = False
        self._built = True

    # ------------------------------------------------------------------
    # Apply  P^{-1}
    # ------------------------------------------------------------------
    def _solve_low_rank(self, x: np.ndarray) -> np.ndarray:
        """
        Apply P⁻¹ to vector x. It implements idea from Frangella, Tropp & Udell (2023),
        adapted to the Schur complement M = μI + S where S ≈ U Λ Uᵀ.

            P⁻¹ x = (λ_k + μ) · U (Λ + μI)⁻¹ Uᵀ x  +  (I - U Uᵀ) x

        where λ_k is the smallest retained eigenvalue of S.
        This is the paper's formula with Ŝ_nys = U Λ Uᵀ playing the role
        of the low-rank approximation to S.

        Quality degrades when μ >> λ_max(S) (identity dominates, sketch irrelevant)
        or when rank is too small to capture the spectrum of S.
        """
        U = self._U
        cache = U.T @ x
        cache *= self.pseudo_inverse
        y = U @ cache
        return x + y

    def __repr__(self):
        cls = type(self).__name__
        return self._get_info(cls)
