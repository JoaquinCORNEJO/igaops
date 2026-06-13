from typing import Literal, Optional, Tuple, List
import logging

from scipy.special import legendre
from scipy import sparse as sp
import numpy as np

from igaops.common import validate_entry
from .utils.template import Template

logger = logging.getLogger(__name__)


class StandardGauss(Template):
    """
    Standard Gauss quadrature class.

    Parameters
    ----------
    degree : int
        polynomial degree of spline
    knotvector : array_like
        knot-vector that defines the spline
    quadtype : {"legendre", "lobatto"}
        type of quadrature, Gauss-Legendre or Gauss-Lobatto
    nbptsperel : int, optional
        Number of quadrature points per element. Default to ``degree + 1``.
    nders : int, optional
        Number of derivatives to stock. Default to 1.
    """

    def __init__(
        self,
        degree: int,
        knotvector: np.ndarray,
        quadtype: Literal["legendre", "lobatto"],
        nbptsperel: Optional[int] = None,
        nders: Optional[int] = None,
    ):
        super().__init__(degree, knotvector)
        self._quadrature_type = validate_entry(quadtype, ["legendre", "lobatto"])
        self._nders = int(nders) if nders is not None else 1

        # Public variables
        self._isoparam_positions: np.ndarray = np.array([])
        self._isoparam_weights: np.ndarray = np.array([])
        self._parametric_weights: np.ndarray = np.array([])
        self._get_isoparametric_quadpts(nbptsperel)
        self._generate()

    @property
    def quadrature_class(self):
        return "GAUSS"

    @property
    def quadrature_type(self) -> Literal["lobatto", "legendre"]:
        return self._quadrature_type

    @property
    def parametric_weights(self):
        "Weights in the reference space [0, 1]"
        return self._parametric_weights

    @property
    def nbptsperel(self):
        "Number of Gauss quadrature points per element"
        return len(self._isoparam_positions)

    def _generate(self):
        if self.quadrature_type == "lobatto" and self.degree == 1:
            # NOTE: De Boor algorithm may not work properly in this case because the
            # derivative of the linear function is discontinuous at breakpoints
            logger.warning(
                "Becarefull, Gauss-Lobatto quadrature is not"
                "supposed to work out with linear polynomials."
            )

        output = self.interpolate_points_and_weights(self.unique_kv)
        self._quadpts = output[0]
        self._parametric_weights = output[1]
        self._set_basis_and_weights()

    def _get_isoparametric_quadpts(self, n_per_el: Optional[int]):
        default = self.degree + (1 if self.quadrature_type == "legendre" else 2)
        n = int(n_per_el) if n_per_el is not None else default
        table = {
            "legendre": np.polynomial.legendre.leggauss,
            "lobatto": lobatto,
        }[self.quadrature_type]
        self._isoparam_positions, self._isoparam_weights = table(n)

    def _set_basis_and_weights(self):
        "Computes the basis and weights of the quadrature class"
        basis = self.eval_basis(self.quadpts, nders=self._nders)
        weights: List[sp.csr_array] = []
        for b in basis:
            w = b.T @ sp.diags(self.parametric_weights)
            weights.extend([w, w])

        # Save data
        self._basis = basis
        self._weights = weights

    def interpolate_points_and_weights(self, knots: np.ndarray):
        "Computes the quadrature points and weights in a set of knots."
        knots = np.unique(knots)
        points = np.concatenate(
            [
                0.5
                * (
                    (knots[i + 1] - knots[i]) * self._isoparam_positions
                    + knots[i]
                    + knots[i + 1]
                )
                for i in range(len(knots) - 1)
            ]
        )
        weights = np.concatenate(
            [
                0.5 * (knots[i + 1] - knots[i]) * self._isoparam_weights
                for i in range(len(knots) - 1)
            ]
        )
        return points, weights


def lobatto(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Lobatto nodes and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points (n >= 2)

    Returns
    -------
    x : ndarray
        Quadrature nodes
    w : ndarray
        Quadrature weights
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    # trivial case
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # Legendre polynomial P_{n-1}
    P = legendre(n - 1)

    # derivative
    dP = np.polyder(P)

    # compute interior roots of P'_{n-1}
    roots = np.roots(dP)
    roots = np.real(roots[np.isreal(roots)])
    roots.sort()

    # assemble nodes
    x = np.empty(n)
    x[0] = -1.0
    x[-1] = 1.0
    x[1:-1] = roots

    # compute weights
    w = np.empty(n)

    # endpoint weights
    w[0] = 2.0 / (n * (n - 1))
    w[-1] = w[0]

    # interior weights
    Px = np.polyval(P, roots)
    w[1:-1] = 2.0 / (n * (n - 1) * Px**2)

    return x, w
