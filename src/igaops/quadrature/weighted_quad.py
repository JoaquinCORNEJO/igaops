from typing import Tuple, Callable, Literal, List, Optional
from dataclasses import dataclass
import logging

from scipy import sparse as sp
import numpy as np

from igaops.common import validate_entry
from .utils.operations import Operations
from .utils.template import Template
from .utils.wq_helpers import (
    compute_quadrature_layout,
    convert_to_greedy_input,
    greedy_min_norm1,
)
from .gauss_quad import StandardGauss as SG

logger = logging.getLogger(__name__)


@dataclass
class _OutputArgs:
    gauss_target: SG
    gsbasis_test: sp.csr_array
    wqbasis_test: Optional[sp.csr_array]
    gsbasis_target: sp.csr_array
    wqbasis_target: Optional[sp.csr_array]


class _PointGenerator:
    @staticmethod
    def endpoint_rule(knots: np.ndarray, n_list: np.ndarray) -> np.ndarray:
        # NOTE: we add 1 because points at knots[0] and knots[-1]
        # are counted as half (they are shared between 2 elements)
        updated = np.asarray([1 + n for n in n_list], dtype=int)
        return _PointGenerator._generate_quadrature_points(knots, updated, np.linspace)

    @staticmethod
    def cellcenter_rule(knots: np.ndarray, n_list: np.ndarray) -> np.ndarray:
        def func(start, end, n):
            return np.array(
                [
                    (1 - (2 * k - 1) / (2 * n)) * start + ((2 * k - 1) / (2 * n)) * end
                    for k in range(1, n + 1)
                ]
            )

        return _PointGenerator._generate_quadrature_points(knots, n_list, func)

    @staticmethod
    def _generate_quadrature_points(
        breakpoints: np.ndarray,
        n_list: np.ndarray,
        algorithm: Callable,
    ) -> np.ndarray:
        quadpts: List[float] = []

        # First span
        tmp = algorithm(breakpoints[0], breakpoints[1], n_list[0])
        quadpts.extend(tmp)

        # Inner spans
        for i in range(1, len(breakpoints) - 2):
            tmp = algorithm(breakpoints[i], breakpoints[i + 1], n_list[i])
            quadpts.extend(tmp)

        # Last span
        tmp = algorithm(breakpoints[-2], breakpoints[-1], n_list[-1])
        quadpts.extend(tmp)

        return np.sort(np.unique(quadpts))


class _Helpers:
    @staticmethod
    def get_deg_kv_target(
        degree_test: int,
        knotvector_test: np.ndarray,
        method: Literal["1", "2"],
    ):
        if method == "1":
            # Space S^[p-1]_[r-1]
            degree_target = degree_test - 1
            knotvector_target = knotvector_test[1:-1]
        elif method == "2":
            # Space S^[p]_[r-1]
            degree_target = degree_test
            knotvector_target = Operations.increase_multiplicity_to_knotvector(
                1, degree_target, knotvector_test
            )
        else:
            raise ValueError("Unknown method for weighted quadrature.")
        return degree_target, knotvector_target

    @staticmethod
    def precompute_data(
        gauss_test: SG,
        method: Literal["1", "2"],
        wq_quadpts: Optional[np.ndarray] = None,
    ) -> _OutputArgs:

        deg_target, kv_target = _Helpers.get_deg_kv_target(
            gauss_test.degree, gauss_test.knotvector, method
        )

        # Basis for W00 in method 1
        B0gs_test = gauss_test.basis[0]
        B0wq_test = None
        if wq_quadpts is not None:
            B0wq_test = gauss_test.eval_basis(wq_quadpts)[0]

        gauss_target = SG(
            degree=deg_target,
            knotvector=kv_target,
            quadtype="legendre",
            nbptsperel=gauss_test.nbptsperel,
        )

        # Basis for W01 in method 1 or W0 in method 2
        B0gs_target = gauss_target.eval_basis(gauss_test.quadpts)[0]
        B0wq_target = None
        if wq_quadpts is not None:
            B0wq_target = gauss_target.eval_basis(wq_quadpts)[0]

        return _OutputArgs(
            gauss_target=gauss_target,
            gsbasis_test=B0gs_test,
            wqbasis_test=B0wq_test,
            gsbasis_target=B0gs_target,
            wqbasis_target=B0wq_target,
        )

    @staticmethod
    def _compute_knot_support(quadpts: np.ndarray) -> np.ndarray:
        quadpts_extended = np.concatenate([[0], quadpts, [1]])
        mean_quapts_extendend = (quadpts_extended[:-1] + quadpts_extended[1:]) / 2.0
        return np.diff(mean_quapts_extendend)

    @staticmethod
    def compute_W0_weights(
        gauss_test: SG,
        quadpts: np.ndarray,
        method: Literal["1", "2"],
        output_args: Optional[_OutputArgs] = None,
    ) -> Tuple[List[sp.csr_array], List[np.ndarray]]:

        if method not in ["1", "2"]:
            raise ValueError("Unknown method for weighted quadrature.")

        # Test space
        W0gs_test = gauss_test.weights[0]
        basis = gauss_test.eval_basis(quadpts, nders=1)

        # Target space
        if output_args is None:
            output_args = _Helpers.precompute_data(
                gauss_test, method, wq_quadpts=quadpts
            )

        # Compute quadrature points support
        quadpts_support = _Helpers._compute_knot_support(quadpts)

        # Compute regularization from support
        regularization = sp.csr_array(basis[0].T @ sp.diags(quadpts_support))

        # Compute the weights
        weights_dense: List[np.ndarray] = []

        def compute(dtype: Literal["test", "target"]):
            "Common computation for method 1 and 2"
            A_op1, A_op2 = output_args.wqbasis_test, output_args.wqbasis_target
            A_mat = A_op1 if dtype == "test" else A_op2
            if A_mat is None:
                raise RuntimeError("Weighted basis are not defined.")

            B_op1, B_op2 = output_args.gsbasis_test, output_args.gsbasis_target
            B_mat = B_op1 if dtype == "test" else B_op2

            integral = sp.csr_array(W0gs_test @ B_mat)
            out = Operations.solve_optimization_problem(
                Z=regularization.toarray(),
                A=A_mat.toarray().T,
                B=integral.toarray(),
            )
            return out

        if method == "1":
            # Computation of W00 for method 1
            weights_dense.append(compute("test"))

        # Computation of W01 for method 1 or W0 for method 2
        weights_dense.append(compute("target"))

        return basis, weights_dense

    @staticmethod
    def compute_W1_weights(p: int, kv: np.ndarray, Q_pm1: np.ndarray) -> np.ndarray:
        # Loop over the functions to construct the weights
        n = len(kv) - p - 1
        out = np.zeros((n, Q_pm1.shape[1]))
        for i in range(n):
            # Term 1
            denom1 = kv[i + p] - kv[i]
            if abs(denom1) > 0 and 0 <= i < len(Q_pm1):
                out[i] += (p / denom1) * Q_pm1[i]

            # Term 2
            denom2 = kv[i + p + 1] - kv[i + 1]
            if abs(denom2) > 0 and 0 <= i + 1 < len(Q_pm1):
                out[i] -= (p / denom2) * Q_pm1[i + 1]

        return out

    @staticmethod
    def eval_nbquadpts_per_element(
        degree: int, knotvector: np.ndarray, quadrature_type: Literal["1", "2"]
    ):
        # Set a minimum
        LOWER = 1

        # Get target degree and knotvector
        deg_target, kv_target = _Helpers.get_deg_kv_target(
            degree_test=degree,
            knotvector_test=knotvector,
            method=quadrature_type,
        )

        # Compute quadrature layout
        S4 = compute_quadrature_layout(
            degree_target=deg_target,
            knot_vector_target=kv_target.tolist(),
            degree_test=degree,
            knot_vector_test=knotvector.tolist(),
        )

        bg, A, b = convert_to_greedy_input(S4)

        # Modify lower bound (best guess)
        bg = np.maximum(bg + 1, LOWER).astype(int)
        b = np.maximum(b + 1, LOWER).astype(int)

        # Perform greedy algorithm
        bg = greedy_min_norm1(bg, A, b)

        return bg.astype(int)


class WeightedQuadrature(Template):
    """
    Weighted quadrature class.

    Parameters
    -----------
    degree : int
        polynomial degree of spline.
    knotvector : array_like
        knot-vector that defines the spline.
    quadtype : {"1", "2"}
        1: first kind based on 4 rules, 2: second kind based on 2 rules.
    position_rule : {"endpoint", "cellcenter"}
        Algorithm's name to compute the position of quadrature points. Default to ``endpoint``.
    """

    def __init__(
        self,
        degree: int,
        knotvector: np.ndarray,
        quadtype: Literal["1", "2"],
        position_rule: Literal["endpoint", "cellcenter"] = "endpoint",
    ):
        super().__init__(degree, knotvector)
        self._quadrature_type = validate_entry(quadtype, ["1", "2"])
        self._position_rule = validate_entry(position_rule, ["endpoint", "cellcenter"])

        self._use_fallback = self.degree <= 1 or len(self.unique_kv) <= 2
        if self._use_fallback:
            logger.warning(
                "Be careful! Weighted quadrature is not supposed to work out"
                "with linear polynomials or one-element knotvectors."
                "By default, Gauss quadrature will be used."
            )

        self._generate()

    @property
    def quadrature_class(self):
        return "WEIGHTED"

    @property
    def quadrature_type(self) -> Literal["1", "2"]:
        return self._quadrature_type

    @property
    def position_rule(self) -> Literal["endpoint", "cellcenter"]:
        return self._position_rule

    def _generate(self):
        if self._use_fallback:
            # Overwrite the quadrature points, basis and weights
            # using standard Gauss quadrature
            q = SG(self.degree, self.knotvector, quadtype="legendre")
            self._quadpts = q.quadpts
            self._weights = q.weights
            self._basis = q.basis
            return
        self._get_parametric_quadpts()
        self._set_basis_and_weights()

    def _get_parametric_quadpts(self):
        n_list = _Helpers.eval_nbquadpts_per_element(
            self.degree, self.knotvector, self.quadrature_type
        )
        PG = _PointGenerator
        if self.position_rule == "endpoint":
            quadpts = PG.endpoint_rule(knots=self.unique_kv, n_list=n_list)
        elif self.position_rule == "cellcenter":
            quadpts = PG.cellcenter_rule(knots=self.unique_kv, n_list=n_list)
        else:
            raise ValueError("Unknown position rule algorithm.")
        self._quadpts = quadpts

    def _set_basis_and_weights(self):
        "Computes the basis and weights of the quadrature class"

        # Cmpute the weights for W0
        gauss_test = SG(self.degree, self.knotvector, quadtype="legendre")

        basis, list_weights = _Helpers.compute_W0_weights(
            gauss_test, self.quadpts, self.quadrature_type
        )

        # To construct W1, we need to compute W0 of the space S^{p-1}_{r-1} for both methods 1 and 2
        # NOTE: It is a coincidence that this space corresponds to the target space of method 1
        gauss_test_tmp = _Helpers.precompute_data(gauss_test, "1").gauss_target

        # Compute the weights for W1
        output_args = _Helpers.precompute_data(
            gauss_test, self.quadrature_type, self.quadpts
        )
        list_weights_tmp = _Helpers.compute_W0_weights(
            gauss_test_tmp,
            self.quadpts,
            self.quadrature_type,
            output_args=output_args,
        )[-1]

        zeros = np.zeros(self.nbquadpts)
        for w in list_weights_tmp:
            # NOTE: we need to add zero weights for the functions that are not in S^{p-1}
            # to be able to apply the combination formula
            w0_mat = np.block([[zeros], [w], [zeros]])
            w1_mat = _Helpers.compute_W1_weights(self.degree, self.knotvector, w0_mat)
            list_weights.append(w1_mat)

        # Build the COO format for the weights
        weights = []
        indices = {"1": [0, 1, 2, 3], "2": [0, 0, 1, 1]}[self.quadrature_type]
        for idx in indices:
            w = sp.csr_array(list_weights[idx])
            w.eliminate_zeros()
            weights.append(w)

        # Save
        self._basis = basis
        self._weights = weights
