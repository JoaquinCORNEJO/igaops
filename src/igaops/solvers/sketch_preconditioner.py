from typing import Optional, Literal, Union

from scipy.sparse.linalg import LinearOperator

from igaops.common import validate_entry
from .sketching.general_preconditioner import RandGeneralPreconditioner
from .sketching.spd_preconditioner import RandSPDPreconditioner


class RandomPreconditioner:
    """
    Preconditioner for the Schur-complement operator

        M = mu I + S

    The preconditioner is built from a low-rank spectral approximation of S
    and then shifted by mu I.

    We assume that the matrix decays rapidly, so a low-rank approximation captures
    its dominant action well, making it effective for spectral preconditioning.

    Construction modes
    ------------------
    Dense (exact)
        Construct S explicitly by applying the operator to the canonical basis.
        Used automatically when m is small or when requested rank >= m.
        Produces an exact factorization.

    Full Randomized (fixed rank)
        Construct a fixed Gaussian sketch Omega, form

            Y = S Omega

        and compute an approximation of the dominant eigenspace.
        Efficient when a suitable target rank is known beforehand.

    Adaptive randomized
        Iteratively construct an orthonormal basis Q spanning the dominant
        range of S using randomized probing, then compute Ritz pairs from

            B = Q^T S Q

        via Rayleigh-Ritz projection.

        This mode automatically determines an effective rank and is often
        more robust than fixed-rank operators.

    The resulting approximate decomposition define the low-rank spectral preconditioner.

    Parameters
    ----------
    S : LinearOperator
        Matvec v -> S @ v.

    mu : float
        Positive spectral shift.

    random_type : {"spd", "general"}
        Type of matrix: Symetric positive definite or General.

    rank : int
        Target rank for randomized approximation.
        If rank is non-positive value, it forces to use an adaptive randomized method.
        Otherwise, it uses a fixed-rank method.

    small_threshold : int
        Threshold below which dense construction is used.
        Set to zero to force randomized preconditioner.

    random_state : int, optional
        Seed for reproducibility.
    """

    registry = {
        "spd": RandSPDPreconditioner,
        "general": RandGeneralPreconditioner,
    }

    def __new__(
        cls,
        S: LinearOperator,
        mu: float,
        random_type: Literal["spd", "general"],
        rank: int = 24,
        small_threshold: int = 576,
        random_state: Optional[int] = None,
    ) -> Union[RandSPDPreconditioner, RandGeneralPreconditioner]:
        validate_entry(random_type, list(cls.registry.keys()))
        subclass = cls.registry[random_type]
        return subclass(S, mu, rank, small_threshold, random_state)
