from typing import List, Any
from abc import ABC, abstractmethod

from scipy import sparse as sp
import numpy as np

from .operations import Operations


class Template(ABC):
    def __init__(self, degree: int, knotvector: np.ndarray):
        # Public variables that need to be checked
        self.degree = degree
        self.knotvector = knotvector

        # Public variables that are fixed
        self._unique_kv = np.unique(np.clip(self.knotvector, a_min=0, a_max=1))
        self._nbctrlpts = len(self.knotvector) - self.degree - 1

        # Public variables that may vary
        self._quadpts: np.ndarray = np.array([])
        self._basis: List[sp.csr_array] = []
        self._weights: List[sp.csr_array] = []

    @property
    def degree(self):
        "Spline degree"
        return self._degree

    @degree.setter
    def degree(self, value):
        if not (np.isscalar(value) and np.real(value) > 0):
            raise ValueError("Degree should be positive integer")
        self._degree = int(np.real(value))

    @property
    def knotvector(self):
        "Spline knotvector"
        return self._knotvector

    @knotvector.setter
    def knotvector(self, value):
        self._knotvector = np.asarray(value).astype(float)

    @property
    def unique_kv(self):
        "Unique knots of knotvector between 0 and 1"
        return self._unique_kv

    @property
    def nbctrlpts(self):
        "Number of control points"
        return self._nbctrlpts

    @property
    def quadpts(self):
        "Quadrature points"
        return self._quadpts

    @quadpts.setter
    def quadpts(self, val):
        if isinstance(val, np.ndarray):
            self._quadpts = val

    @property
    def basis(self):
        "Quadrature basis"
        return self._basis

    @basis.setter
    def basis(self, val):
        if len(val) > 0 and all(sp.issparse(v) for v in val):
            self._basis = val

    @property
    def weights(self):
        "Quadrature weights"
        return self._weights

    @weights.setter
    def weights(self, val):
        if len(val) > 0 and all(sp.issparse(v) for v in val):
            self._weights = val

    @property
    def nbquadpts(self):
        "Number of quadrature points"
        return len(self.quadpts)

    @property
    def max_h_size(self) -> float:
        "Maximal size among knot-spans"
        return float(np.max(np.abs(np.diff(self.unique_kv))))

    @property
    @abstractmethod
    def quadrature_class(self) -> Any:
        "Quadrature class"
        raise NotImplementedError("To implement in children")

    @property
    @abstractmethod
    def quadrature_type(self) -> Any:
        "Quadrature type"
        raise NotImplementedError("To implement in children")

    @abstractmethod
    def _generate(self):
        "Initializes the quadrature class"
        raise NotImplementedError("To implement in children")

    def eval_basis(self, knots_to_interp: np.ndarray, nders: int = 1):
        """
        Evaluates basis functions and its first derivative at given knots.

        Parameters
        ----------
        knots_to_interp : np.ndarray
            The list of knots where the spline is evaluated.
        nders : int, optional
            Number of derivatives to compute. Defaults to 1.

        Returns
        -------
        List[sparray]
            List of basis and derivatives.
        """
        return Operations.eval_ders_basis_sparse(
            self.degree,
            self.knotvector,
            knots_to_interp,
            nders=nders,
        )

    @classmethod
    def create_minimal(
        cls, degree: int, knotvector: np.ndarray, knots: np.ndarray, nders: int = 1
    ):
        instance = cls.__new__(cls)
        Template.__init__(
            instance,
            degree=degree,
            knotvector=knotvector,
        )
        basis = instance.eval_basis(knots, nders=nders)
        instance.quadpts = knots
        instance.basis = basis
        instance.weights = []
        return instance

    def __repr__(self) -> str:
        message = f""""
            \n {self.quadrature_class} QUADRATURE:
            type: {self.quadrature_type}
            degree: {self.degree}
            nb of functions: {self.nbctrlpts}
            max knot-span: {self.max_h_size:.2e}
            nb of quadrature points: {self.nbquadpts}
        """
        return message
