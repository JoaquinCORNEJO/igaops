from typing import Sequence, Literal, Union, List, Optional

from scipy import sparse as sp
import numpy as np

from igaops.quadrature import StandardGauss, WeightedQuadrature
from .utils.template import Template
from .bsplines_ops import BsplineOperations, matvec_customized

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


class _Helpers:
    @staticmethod
    def project_nurbs_weights(
        basis_list: Sequence[List[sp.csr_array]],
        nurbs_weights: np.ndarray,
        nders: int = 0,
        along_axis: bool = False,
    ) -> Sequence[np.ndarray]:
        """
        Compute the projection of the NURBS weights and its derivatives
        on the quadrature points. Returns w_proj, z_proj, h_proj.
        """
        nurbs_weights = np.atleast_2d(nurbs_weights)
        inv_weights_proj = np.array([])
        jac_weights_proj = np.array([])
        hes_weights_proj = np.array([])

        if nders >= 0:
            weights_proj = BsplineOperations.eval_interpolation(
                basis_list, nurbs_weights, along_axis=along_axis
            )[0]
            inv_weights_proj = 1.0 / weights_proj
        if nders >= 1:
            jac_weights_proj = BsplineOperations.eval_jacobien(
                basis_list, nurbs_weights, along_axis=along_axis
            )[0]
        if nders >= 2:
            hes_weights_proj = BsplineOperations.eval_hessian(
                basis_list, nurbs_weights, along_axis=along_axis
            )[0]

        if nders == 0:
            return inv_weights_proj, np.array([]), np.array([])

        if nders == 1:
            return (
                inv_weights_proj,
                jac_weights_proj * inv_weights_proj**2,
                np.array([]),
            )

        if nders == 2:
            return (
                inv_weights_proj,
                jac_weights_proj * inv_weights_proj**2,
                hes_weights_proj * inv_weights_proj**3,
            )
        raise NotImplementedError(
            "Number of derivatives should be positive and lower than 3"
        )

    @staticmethod
    def apply_scaling(
        array_in: np.ndarray,
        weights: np.ndarray,
        left_depth: int = 0,
        right_depth: int = 0,
    ) -> np.ndarray:
        """
        A broadcasting version of *-product between a n-rank tensor and 1-rank array.
        In this class it is used with depth=0, 1, 2.
        """
        left_indices = [np.newaxis] * left_depth
        right_indices = [np.newaxis] * right_depth
        return weights[*left_indices, :, *right_indices] * array_in


class NurbsOperations(Template):
    @staticmethod
    def eval_hessian(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:
        ndim = len(basis_list)
        w_proj, z_proj, h_proj = _Helpers.project_nurbs_weights(
            basis_list, nurbs_weights, nders=2, along_axis=along_axis
        )
        u_at_ctrlpts_scaled = _Helpers.apply_scaling(
            u_at_ctrlpts, nurbs_weights, left_depth=1
        )
        hess_first = BsplineOperations.eval_hessian(
            basis_list, u_at_ctrlpts_scaled, along_axis=along_axis
        )
        hess_first = _Helpers.apply_scaling(hess_first, w_proj, left_depth=3)
        hess = np.zeros_like(hess_first, dtype=float)

        for i in range(hess.shape[0]):
            for j in range(hess.shape[1]):
                alpha_list = np.zeros(ndim, dtype=int)
                alpha_list[j] = 1
                for k in range(j, hess.shape[2]):
                    beta_list = np.zeros(ndim, dtype=int)
                    beta_list[k] = 1
                    hess_second_1_ijk = (
                        matvec_customized(
                            [basis[a] for basis, a in zip(basis_list, alpha_list)],
                            u_at_ctrlpts_scaled[i],
                            along_axis=along_axis,
                        )
                        * z_proj[k]
                    )
                    hess_second_2_ijk = (
                        matvec_customized(
                            [basis[b] for basis, b in zip(basis_list, beta_list)],
                            u_at_ctrlpts_scaled[i],
                            along_axis=along_axis,
                        )
                        * z_proj[j]
                    )
                    u_projected = matvec_customized(
                        [basis[0] for basis in basis_list],
                        u_at_ctrlpts_scaled[i],
                        along_axis=along_axis,
                    )
                    hess_third_1_ij = u_projected * z_proj[j] * z_proj[k] / w_proj
                    hess_third_2_ij = u_projected * h_proj[j][k] / w_proj
                    hess[i][j][k] = hess[i][k][j] = (
                        hess_first[i][j][k]
                        - hess_second_1_ijk
                        - hess_second_2_ijk
                        + 2 * hess_third_1_ij
                        - hess_third_2_ij
                    )
        return hess

    @staticmethod
    def eval_jacobien(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:
        w_proj, z_proj, _ = _Helpers.project_nurbs_weights(
            basis_list, nurbs_weights, nders=1, along_axis=along_axis
        )
        u_at_ctrlpts_scaled = _Helpers.apply_scaling(
            u_at_ctrlpts, nurbs_weights, left_depth=1
        )
        jac_first = BsplineOperations.eval_jacobien(
            basis_list, u_at_ctrlpts_scaled, along_axis=along_axis
        )
        jac_first = _Helpers.apply_scaling(jac_first, w_proj, left_depth=2)
        jac = np.zeros_like(jac_first, dtype=float)
        for i in range(jac.shape[0]):
            for j in range(jac.shape[1]):
                jac_second_ij = (
                    matvec_customized(
                        [basis[0] for basis in basis_list],
                        u_at_ctrlpts_scaled[i],
                        along_axis=along_axis,
                    )
                    * z_proj[j]
                )
                jac[i][j] = jac_first[i][j] - jac_second_ij
        return jac

    @staticmethod
    def eval_interpolation(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:
        w_proj = _Helpers.project_nurbs_weights(
            basis_list, nurbs_weights, nders=0, along_axis=along_axis
        )[0]
        u_at_ctrlpts_scaled = _Helpers.apply_scaling(
            u_at_ctrlpts, nurbs_weights, left_depth=1
        )
        u_interp = BsplineOperations.eval_interpolation(
            basis_list, u_at_ctrlpts_scaled, along_axis=along_axis
        )
        return _Helpers.apply_scaling(u_interp, w_proj, left_depth=1)

    @staticmethod
    def assemble_scalar_u_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def assemble_scalar_gradu_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def assemble_scalar_u_force(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        w_proj = _Helpers.project_nurbs_weights(
            [q.basis for q in quadrule_list], nurbs_weights, nders=0
        )[0]
        coefficients_copy = _Helpers.apply_scaling(coefficients, w_proj, left_depth=1)
        array_out = BsplineOperations.assemble_scalar_u_force(
            quadrule_list, coefficients_copy
        )
        return _Helpers.apply_scaling(array_out, nurbs_weights, left_depth=1)

    @staticmethod
    def assemble_scalar_gradu_force(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        w_proj, z_proj, _ = _Helpers.project_nurbs_weights(
            [q.basis for q in quadrule_list], nurbs_weights, nders=1
        )
        # First term
        coefficients_copy = _Helpers.apply_scaling(coefficients, w_proj, left_depth=2)
        array_out = BsplineOperations.assemble_scalar_gradu_force(
            quadrule_list, coefficients_copy
        )
        # Second term
        coefficients_copy = np.einsum(
            "ij...,j...->i...", coefficients, z_proj, optimize=True
        )
        array_out -= BsplineOperations.assemble_scalar_u_force(
            quadrule_list, coefficients_copy
        )
        return _Helpers.apply_scaling(array_out, nurbs_weights, left_depth=1)

    @staticmethod
    def compute_mf_scalar_u_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        rd = array_in.ndim - 1
        w_proj = _Helpers.project_nurbs_weights(
            [q.basis for q in quadrule_list], nurbs_weights, nders=0
        )[0]
        array_in_copy = _Helpers.apply_scaling(
            array_in, nurbs_weights, left_depth=0, right_depth=rd
        )
        coefficients_copy = _Helpers.apply_scaling(
            coefficients, w_proj**2, left_depth=0
        )
        array_out = BsplineOperations.compute_mf_scalar_u_v(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        return _Helpers.apply_scaling(
            array_out, nurbs_weights, left_depth=0, right_depth=rd
        )

    @staticmethod
    def compute_mf_scalar_gradu_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        enable_spacetime = time_ders is not None
        rd = array_in.ndim - 1
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim
        w_proj, z_proj, _ = _Helpers.project_nurbs_weights(
            [q.basis for q in quadrule_list], nurbs_weights, nders=1
        )
        z_proj = z_proj[:sp_ndim, ...]
        array_in_copy = _Helpers.apply_scaling(
            array_in, nurbs_weights, left_depth=0, right_depth=rd
        )
        # First term
        coefficients_copy = _Helpers.apply_scaling(
            coefficients, w_proj**2, left_depth=2
        )
        array_out = BsplineOperations.compute_mf_scalar_gradu_gradv(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        # Second term
        coefficients_copy = np.einsum(
            "ij...,i...,j...->...", coefficients, z_proj, z_proj, optimize=True
        )
        array_out += BsplineOperations.compute_mf_scalar_u_v(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        # Third term
        coefficients_copy = np.einsum(
            "ij...,i...,...->j...", coefficients, z_proj, w_proj, optimize=True
        )
        array_out -= BsplineOperations.compute_mf_scalar_u_gradv(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        # Fourth term
        coefficients_copy = np.einsum(
            "ij...,j...,...->i...", coefficients, z_proj, w_proj, optimize=True
        )
        array_out -= BsplineOperations.compute_mf_scalar_gradu_v(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        return _Helpers.apply_scaling(
            array_out, nurbs_weights, left_depth=0, right_depth=rd
        )

    @staticmethod
    def _compute_mf_scalar_gradu_v_or_u_grad_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
        mf_type: Literal["gradu_v", "u_gradv"] = "gradu_v",
    ) -> np.ndarray:
        enable_spacetime = time_ders is not None
        rd = array_in.ndim - 1
        ndim = len(quadrule_list)
        sp_ndim = ndim - 1 if enable_spacetime else ndim
        w_proj, z_proj, _ = _Helpers.project_nurbs_weights(
            [q.basis for q in quadrule_list], nurbs_weights, nders=1
        )
        z_proj = z_proj[:sp_ndim, ...]
        array_in_copy = _Helpers.apply_scaling(
            array_in, nurbs_weights, left_depth=0, right_depth=rd
        )
        oper = (
            BsplineOperations.compute_mf_scalar_gradu_v
            if mf_type == "gradu_v"
            else BsplineOperations.compute_mf_scalar_u_gradv
        )
        # First term
        coefficients_copy = _Helpers.apply_scaling(
            coefficients, w_proj**2, left_depth=1
        )
        array_out = oper(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        # Second term
        coefficients_copy = np.einsum(
            "i...,i...,...->...", coefficients, z_proj, w_proj, optimize=True
        )
        array_out -= BsplineOperations.compute_mf_scalar_u_v(
            quadrule_list,
            coefficients_copy,
            array_in_copy,
            time_ders=time_ders,
        )
        return _Helpers.apply_scaling(
            array_out, nurbs_weights, left_depth=0, right_depth=rd
        )

    @staticmethod
    def compute_mf_scalar_gradu_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        return NurbsOperations._compute_mf_scalar_gradu_v_or_u_grad_v(
            quadrule_list,
            coefficients,
            array_in,
            time_ders,
            nurbs_weights,
            mf_type="gradu_v",
        )

    @staticmethod
    def compute_mf_scalar_u_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        return NurbsOperations._compute_mf_scalar_gradu_v_or_u_grad_v(
            quadrule_list,
            coefficients,
            array_in,
            time_ders,
            nurbs_weights,
            mf_type="u_gradv",
        )
