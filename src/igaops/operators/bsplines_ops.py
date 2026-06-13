from typing import Sequence, Literal, Union, List, Optional

from scipy import sparse as sp
import numpy as np

from igaops.common import KronProduct
from igaops.quadrature import StandardGauss, WeightedQuadrature
from .utils.matrix_free import MatrixFree
from .utils.template import Template
from .utils.operations import Operations

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


def matvec_customized(
    matrix_list: Sequence[sp.csr_array], vector: np.ndarray, along_axis: bool
):
    if not along_axis:
        return MatrixFree.apply(
            matrix_list,
            vector,
            is_transpose=False,
        )

    # FIXME: poor performance due to dense matrices
    dense_list = [m.toarray() for m in matrix_list]
    return KronProduct.mf_kron_following_axis(dense_list, vector, axis=1)


class _Helpers:
    @staticmethod
    def get_nbofrows(
        quadrule_list: Sequence[QuadratureRule],
    ) -> np.ndarray:
        return np.asarray([q.nbctrlpts for q in quadrule_list], dtype=int)

    @staticmethod
    def get_nbofcols(basis_list: Sequence[List[sp.csr_array]]) -> np.ndarray:
        return np.asarray([b[0].shape[0] for b in basis_list], dtype=int)

    @staticmethod
    def build_matrix_list(
        quadrule_list: Sequence[QuadratureRule],
        idx_list: Sequence[int],
        product_type: Literal["basis", "weights"],
    ) -> Sequence[sp.csr_array]:
        if product_type == "basis":
            matrix_list = [q.basis[idx] for q, idx in zip(quadrule_list, idx_list)]
        elif product_type == "weights":
            matrix_list = [q.weights[idx] for q, idx in zip(quadrule_list, idx_list)]
        else:
            raise ValueError("Product type should be basis or weights")
        return matrix_list


class BsplineOperations(Template):
    @staticmethod
    def eval_hessian(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:

        # Number of scalar fields
        nm = np.shape(u_at_ctrlpts)[0]
        # Number of parameteric variables xi = (xi_1, xi_2, ...)
        ndim = len(basis_list)

        nc_list = _Helpers.get_nbofcols(basis_list)
        nc = max(nc_list) if along_axis else np.prod(nc_list)
        hess = np.zeros((nm, ndim, ndim, nc))  # H_ijk = d^2 u_i / d xi_j / d xi_k
        for i in range(nm):
            for j in range(ndim):
                alpha_list = np.zeros(ndim, dtype=int)
                alpha_list[j] = 1
                for k in range(j, ndim):
                    beta_list = np.zeros(ndim, dtype=int)
                    beta_list[k] = 1
                    zeta_list = alpha_list + beta_list
                    hess[i][j][k] = hess[i][k][j] = matvec_customized(
                        [basis[zeta] for basis, zeta in zip(basis_list, zeta_list)],
                        u_at_ctrlpts[i],
                        along_axis=along_axis,
                    )
        return hess

    @staticmethod
    def eval_jacobien(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:

        # Number of scalar fields
        nm = np.shape(u_at_ctrlpts)[0]
        # Number of parameteric variables xi = (xi_1, xi_2, ...)
        ndim = len(basis_list)

        nc_list = _Helpers.get_nbofcols(basis_list)
        nc = max(nc_list) if along_axis else np.prod(nc_list)
        jac = np.zeros((nm, ndim, nc))  # J_ij = d u_i / d xi_j
        for i in range(nm):
            for j in range(ndim):
                beta_list = np.zeros(ndim, dtype=int)
                beta_list[j] = 1
                jac[i][j] = matvec_customized(
                    [basis[beta] for basis, beta in zip(basis_list, beta_list)],
                    u_at_ctrlpts[i],
                    along_axis=along_axis,
                )

        return jac

    @staticmethod
    def eval_interpolation(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:

        # Number of scalar fields
        nm = np.shape(u_at_ctrlpts)[0]
        nc_list = _Helpers.get_nbofcols(basis_list)
        nc = max(nc_list) if along_axis else np.prod(nc_list)
        u_interp = np.zeros((nm, nc))
        for i in range(nm):
            u_interp[i] = matvec_customized(
                [basis[0] for basis in basis_list],
                u_at_ctrlpts[i],
                along_axis=along_axis,
            )

        return u_interp

    @staticmethod
    def assemble_scalar_u_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> sp.csr_array:
        ndim = len(quadrule_list)
        zero_list = np.zeros(ndim, dtype=int).tolist()
        basis = _Helpers.build_matrix_list(
            quadrule_list, zero_list, product_type="basis"
        )
        weights = _Helpers.build_matrix_list(
            quadrule_list, zero_list, product_type="weights"
        )
        return Operations.sumfact_assembling(weights, basis, coefficients)

    @staticmethod
    def assemble_scalar_gradu_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> sp.csr_array:
        ndim = len(quadrule_list)
        nr_list = _Helpers.get_nbofrows(quadrule_list)
        matrix = sp.csr_array((np.prod(nr_list), np.prod(nr_list)))
        for j in range(ndim):
            beta_list = np.zeros(ndim, dtype=int)
            beta_list[j] = 1
            basis = _Helpers.build_matrix_list(
                quadrule_list, beta_list.tolist(), "basis"
            )
            for i in range(ndim):
                alpha_list = np.zeros(ndim, dtype=int)
                alpha_list[i] = 1
                zeta_list = beta_list + 2 * alpha_list
                weights = _Helpers.build_matrix_list(
                    quadrule_list, zeta_list.tolist(), "weights"
                )
                matrix += Operations.sumfact_assembling(
                    weights, basis, coefficients[i][j]
                )
        return matrix

    @staticmethod
    def assemble_scalar_u_force(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        nm, _ = np.shape(coefficients)
        nr_list = _Helpers.get_nbofrows(quadrule_list)
        array_out = np.zeros((nm, np.prod(nr_list)))
        for i in range(nm):
            array_out[i] = MatrixFree.apply(
                [q.weights[0] for q in quadrule_list],
                coefficients[i],
                is_transpose=False,
            )
        return array_out

    @staticmethod
    def assemble_scalar_gradu_force(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        ndim = len(quadrule_list)
        nm = np.shape(coefficients)[0]
        nr_list = _Helpers.get_nbofrows(quadrule_list)
        array_out = np.zeros((nm, np.prod(nr_list)))
        for j in range(nm):
            for i in range(ndim):
                zeta_list = np.zeros(ndim, dtype=int)
                zeta_list[i] = 2
                array_out += MatrixFree.apply(
                    [q.weights[zeta] for q, zeta in zip(quadrule_list, zeta_list)],
                    coefficients[j][i],
                    is_transpose=False,
                )
        return array_out

    @staticmethod
    def compute_mf_scalar_u_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:

        enable_spacetime = time_ders is not None
        ndim = len(quadrule_list)
        zero_list = np.zeros(ndim, dtype=int)
        if enable_spacetime:
            zero_list[-1] = time_ders[1]
        array_tmp = MatrixFree.apply(
            [q.basis[beta] for q, beta in zip(quadrule_list, zero_list)],
            array_in,
            is_transpose=False,
        )

        if enable_spacetime:
            zero_list[-1] = time_ders[0]
        broadcast = np.einsum("i,i...->i...", coefficients, array_tmp, optimize=True)
        array_out = MatrixFree.apply(
            [q.weights[alpha] for q, alpha in zip(quadrule_list, zero_list)],
            broadcast,
            is_transpose=False,
        )
        return array_out

    @staticmethod
    def compute_mf_scalar_gradu_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        enable_spacetime = time_ders is not None
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim
        array_out = np.zeros_like(array_in, dtype=float)
        for j in range(sp_ndim):
            beta_list = np.zeros(ndim, dtype=int)
            beta_list[j] = 1
            if enable_spacetime:
                beta_list[-1] = time_ders[1]
            array_tmp = MatrixFree.apply(
                [q.basis[beta] for q, beta in zip(quadrule_list, beta_list)],
                array_in,
                is_transpose=False,
            )
            broadcast = np.einsum(
                "lm,m...->lm...", coefficients[:, j, :], array_tmp, optimize=True
            )
            for i in range(sp_ndim):
                alpha_list = np.zeros(ndim, dtype=int)
                alpha_list[i] = 1
                if enable_spacetime:
                    alpha_list[-1] = time_ders[0]
                zeta_list = beta_list + 2 * alpha_list
                array_out += MatrixFree.apply(
                    [q.weights[zeta] for q, zeta in zip(quadrule_list, zeta_list)],
                    broadcast[i],
                    is_transpose=False,
                )
        return array_out

    @staticmethod
    def compute_mf_scalar_gradu_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        enable_spacetime = time_ders is not None
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim
        array_out = np.zeros_like(array_in, dtype=float)

        beta_list = np.zeros(ndim, dtype=int)
        if enable_spacetime:
            beta_list[-1] = time_ders[1]
        array_tmp = MatrixFree.apply(
            [q.basis[beta] for q, beta in zip(quadrule_list, beta_list)],
            array_in,
            is_transpose=False,
        )
        broadcast = np.einsum("ij,j...->ij...", coefficients, array_tmp, optimize=True)
        for i in range(sp_ndim):
            alpha_list = np.zeros(ndim, dtype=int)
            alpha_list[i] = 1
            if enable_spacetime:
                alpha_list[-1] = time_ders[0]
            zeta_list = beta_list + 2 * alpha_list
            array_out += MatrixFree.apply(
                [q.weights[zeta] for q, zeta in zip(quadrule_list, zeta_list)],
                broadcast[i],
                is_transpose=False,
            )
        return array_out

    @staticmethod
    def compute_mf_scalar_u_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        enable_spacetime = time_ders is not None
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim

        nbcols = np.asarray([q.nbquadpts for q in quadrule_list], dtype=int)
        array_tmp = np.zeros((np.prod(nbcols), *array_in.shape[1:]))
        for i in range(sp_ndim):
            beta_list = np.zeros(ndim, dtype=int)
            beta_list[i] = 1
            if enable_spacetime:
                beta_list[-1] = time_ders[1]
            array_in_interp = MatrixFree.apply(
                [q.basis[beta] for q, beta in zip(quadrule_list, beta_list)],
                array_in,
                is_transpose=False,
            )
            broadcast = np.einsum(
                "i,i...->i...", coefficients[i], array_in_interp, optimize=True
            )
            array_tmp += broadcast

        alpha_list = np.zeros(ndim, dtype=int)
        if enable_spacetime:
            alpha_list[-1] = time_ders[0]
        array_out = MatrixFree.apply(
            [q.weights[alpha] for q, alpha in zip(quadrule_list, alpha_list)],
            array_tmp,
            is_transpose=False,
        )
        return array_out
