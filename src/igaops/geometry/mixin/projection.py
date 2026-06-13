from typing import Tuple, Optional, Literal, List, Any
from dataclasses import dataclass
import logging

import numpy as np

from igaops.common import Constants, BoundarySide, ParametricDirection, validate_entry
from igaops.fastdiag import SingleFD
from igaops.solvers import NonLinearSolver, LinearSolver

from .template import SingleSpline
from .operations import Operations
from .reader import Reader

logger = logging.getLogger(__name__)


class ProjectionMixin(SingleSpline):
    def interpolate_field(
        self,
        u_at_ctrlpts: np.ndarray,
        knots_list: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Project the field U from control points to quadrature points.
        """
        if u_at_ctrlpts.ndim != 2:
            raise ValueError("Expected 2D array.")

        if knots_list is not None:
            basis_list = [
                q.eval_basis(k) for q, k in zip(self.quadrule_list, knots_list)
            ]
        else:
            basis_list = [q.basis for q in self.quadrule_list]

        return self.operator_engine.eval_interpolation(
            basis_list=basis_list,
            u_at_ctrlpts=u_at_ctrlpts,
            nurbs_weights=self.nurbs_weights,
        )

    def interpolate_jacobien_field(
        self,
        u_at_ctrlpts: np.ndarray,
        knots_list: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Project the jacobian of the field U from control points to quadrature points.
        """
        if u_at_ctrlpts.ndim != 2:
            raise ValueError("Expected 2D array.")

        if knots_list is not None:
            basis_list = [
                q.eval_basis(k) for q, k in zip(self.quadrule_list, knots_list)
            ]
        else:
            basis_list = [q.basis for q in self.quadrule_list]

        return self.operator_engine.eval_jacobien(
            basis_list=basis_list,
            u_at_ctrlpts=u_at_ctrlpts,
            nurbs_weights=self.nurbs_weights,
        )

    def integrate_field_over_domain(
        self, u_at_quadpts: np.ndarray, mask_at_quadpts: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes the integral of function u (evaluated at quadrature points) over the domain.
        """

        if u_at_quadpts.ndim != 2:
            raise ValueError("Expected 2D array.")

        # Recover determinant
        spline = self.evaluate_spline()
        determinant = spline.det_jac

        new_u_at_quadpts = u_at_quadpts.copy()
        if mask_at_quadpts is not None:
            nr = u_at_quadpts.shape[0]
            nc = len(determinant)
            new_u_at_quadpts = np.zeros((nr, nc))
            new_u_at_quadpts[:, mask_at_quadpts] = u_at_quadpts

        # Compute the integral of the field over the domain
        coefficients: np.ndarray = new_u_at_quadpts * determinant[np.newaxis, :]
        return self.operator_engine.assemble_scalar_u_force(
            self.quadrule_list, coefficients, nurbs_weights=self.nurbs_weights
        )

    def solve_gram_system(
        self,
        rhs: np.ndarray,
        tolerance: float = Constants.TINY,
        maxiters: int = 20,
    ) -> np.ndarray:
        """
        Project the field U from quadrature points to control points.
        """

        if rhs.ndim != 2:
            raise ValueError("Expected 2D array.")

        # Recover determinant
        spline = self.evaluate_spline()
        determinant = spline.det_jac

        # Define matrix-vector operation
        def matvec(x_in):
            x_out = self.operator_engine.compute_mf_scalar_u_v(
                self.quadrule_list,
                determinant,
                x_in,
                nurbs_weights=self.nurbs_weights,
            )
            return x_out

        # Define preconditioner
        fastdiag = SingleFD()
        fastdiag.compute_space_eigendecomposition(
            self.quadrule_list, np.zeros((1, self.ndim, 2))
        )
        fastdiag.update_space_eigenvalues(scalar_coefs=[1, 0])
        precond = fastdiag.apply_spatial_preconditioner

        # Solve projection problem
        # TODO: introduce "block_cg" instead of "cg"
        linear_solver = LinearSolver(
            tolerance=tolerance, maxiters=maxiters, linear_type="cg", verbose=False
        )
        array_out = []
        for arr in rhs:
            output = linear_solver.solve(matvec, arr, precond)
            array_out.append(output.solution)

        return np.vstack(array_out)

    def closest_point_projection(
        self,
        X_target: np.ndarray,
        localization: Tuple[ParametricDirection, BoundarySide],
        tolerance: float = Constants.TINY,
        method: Literal["picard", "newton"] = "newton",
        warping: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the parametric coordinates corresponding to a target physical point on a specified boundary.

        Parameters
        ----------
        X_target : ndarray
            Target physical coordinates, shape (spatial_dim, n_points).
        localization : Tuple[ParametricDirection, BoundarySide]
            Tuple specifying the parametric direction and boundary side.
        tolerance : float
            Threshold for the convergence criteria in nonlinear solver.
        method : {"picard", "newton"}
            Picard is a fixed point algorithm whereas Newton computes consistent tangent matrix,
            (it requires that the second derivative of splines are defined).
        warping : ndarray, optional
            Displacement offset to add to control points. Default None.

        Raises
        ------
        RuntimeError
            If the nonlinear solver fails to converge or if the solution is out of bounds.
        """
        if self.ndim < 2:
            raise NotImplementedError("Expected surface or volume.")

        if X_target.ndim != 2:
            raise ValueError("Expected 2D array.")

        # Prepate inputs for computations
        X_target = Reader.ensure_3_rows(X_target)
        projector = _ClosestPointProjector(self, localization, method=method)

        # Set first guess from localization data
        solution = projector.generate_guess(X_target.shape[1])

        # Use projector class for solving
        projector.solve(X_target, solution, warping, tolerance=tolerance)
        status, mask = projector.get_status(solution)

        if status == _Status.FAILED:
            raise RuntimeError("Close projection failed.")

        solution = projector.reconstruct(solution)

        return solution, mask


class _Status:
    """
    Tells if Closest Point algorithm converged,
    failed or the solution is out of bounds.
    """

    CONVERGED = 1
    OUTBOUND = 0
    FAILED = -1


@dataclass
class _ProjectorVars:
    """
    Save variables for projection algorithm.
    """

    X_target: np.ndarray
    warping: np.ndarray
    tangent: Optional[np.ndarray]


class _ClosestPointProjector:

    nonlinearsolver = NonLinearSolver(
        tolerance=Constants.TINY,
        maxiters=20,
        allow_acceleration=True,
        allow_line_search=False,
        verbose=True,
    )

    def __init__(
        self,
        patch: ProjectionMixin,
        localization: Tuple[ParametricDirection, BoundarySide],
        method: Literal["picard", "newton"] = "newton",
    ):
        self.patch = patch
        self.localization = localization
        self.method = validate_entry(method, ["picard", "newton"])

        self.quadrature_list = [
            self.patch.quadrule_list[i]
            for i in range(self.patch.ndim)
            if ParametricDirection(i) != self.localization[0]
        ]

        self._projector_vars: Optional[_ProjectorVars] = None
        self._last_simulation: Optional[Any] = None

    @property
    def projector_vars(self):
        if self._projector_vars is None:
            raise RuntimeError("Projector variable is undefined.")
        return self._projector_vars

    def _clip_0_to_1(self, array: np.ndarray):
        """
        Force the values of arrays to be between 0 and 1.

        NOTE
        ----
        Clipping is necessary for convergence in some cases.
        """
        array[:] = np.clip(array, -Constants.TINY, 1.0 + Constants.TINY)

    def _eval_projection(self, solution: np.ndarray, compute_hessian: bool):
        """
        Evaluates geometry, Jacobian and optionally Hessian.
        """
        # Get indices
        indices = self.patch.boundary_manager.select(self.localization)

        # Select control points and weights on boundary
        ctrlpts = self.patch.ctrlpts + self.projector_vars.warping
        ctrlpts = ctrlpts[:, indices]
        nurbs_weights = (
            self.patch.nurbs_weights[indices]
            if self.patch.nurbs_weights.size > 0
            else np.array([])
        )

        # Computations
        nders = 2 if compute_hessian else 1
        basis_list = [
            q.eval_basis(k, nders=nders) for q, k in zip(self.quadrature_list, solution)
        ]

        X = self.patch.operator_engine.eval_interpolation(
            basis_list, ctrlpts, nurbs_weights=nurbs_weights, along_axis=True
        )  # (x_i)

        J = self.patch.operator_engine.eval_jacobien(
            basis_list, ctrlpts, nurbs_weights=nurbs_weights, along_axis=True
        )  # (d x_i / d xi_j)

        H = None
        if compute_hessian:
            H = self.patch.operator_engine.eval_hessian(
                basis_list, ctrlpts, nurbs_weights=nurbs_weights, along_axis=True
            )  # (d^2 x_i / d xi_j /d xi_k)

        return X, J, H

    def _compute_residual(
        self, solution: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute the gradient of the square-distance problem.

        The residual is:
        r = J^T (X_target - X)
        """

        # Preambule
        X_target = self.projector_vars.X_target

        compute_hessian = (
            self.method == "newton" and np.min(list(self.patch.degree)) > 1
        )

        self._clip_0_to_1(solution)

        # Computations
        X_k, J_k, H_k = self._eval_projection(solution, compute_hessian)

        res_k = np.einsum("ik,ijk->jk", X_target - X_k, J_k, optimize=True)
        tan_k = np.einsum("lik,ljk->ijk", J_k, J_k, optimize=True)

        if H_k is not None:
            tan_k -= np.einsum("il,ijkl->jkl", X_target - X_k, H_k, optimize=True)

        self.projector_vars.tangent = tan_k
        return res_k, {}

    def _compute_increment(self, residual: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the Newton/Picard increment.
        """
        tangent = self.projector_vars.tangent
        if tangent is None:
            raise RuntimeError("Tangent is undefined.")
        _, invtang = Operations.eval_inverse_and_determinant(tangent)
        incr = np.einsum("ijk,jk->ik", invtang, residual, optimize=True)
        return incr

    def generate_guess(self, nnz: int):
        """
        Generate the default initial guess xi = 0.5 in the active directions.
        """
        return np.full((self.patch.ndim - 1, nnz), 0.5)

    def reconstruct(self, array: np.ndarray):
        """
        Generate the true solution (with the correct dimensionality).
        """
        ndim = self.patch.ndim
        direction, side = self.localization
        indices = [i for i in range(ndim) if ParametricDirection(i) != direction]
        values = 0.0 if side == BoundarySide.MIN else 1.0
        complete = np.full((ndim, array.shape[1]), values)
        complete[indices] = array
        return complete

    def solve(
        self,
        X_target: np.ndarray,
        solution: np.ndarray,
        warping: Optional[np.ndarray] = None,
        tolerance: float = Constants.TINY,
    ):
        """
        Solves the closes-point problem.

        Parameters
        ----------
        X_target : array_like
            The target physical points.
        solution : array_like
            Initial parametric coordinates. Modified in-situ.
        warping : array_like
            Optional displacement / warping of the control points.
        tolerance : float
            Residual relative tolerance.
        """

        if warping is None:
            warping = np.zeros_like(self.patch.ctrlpts)

        self._projector_vars = _ProjectorVars(
            X_target=X_target, warping=warping, tangent=None
        )
        self.nonlinearsolver.update(
            tolerance=tolerance, allow_acceleration=self.method == "picard"
        )
        self._last_simulation = self.nonlinearsolver.solve(
            solution,
            compute_residual=self._compute_residual,
            compute_increment=self._compute_increment,
        )
        self._projector_vars = None

    def get_status(self, array_in: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Get status after running closest-point algorithm:
        """

        if self._last_simulation is None:
            logger.warning("The last simulation is not stored.")
            return _Status.FAILED, np.array([])

        # Evaluate residual
        res_success = self._last_simulation.success

        if not res_success:
            logger.warning("Nonlinear solver did not converge.")
            return _Status.FAILED, np.array([])

        # Evaluate points out of bounds
        array_to_bool = np.all(
            (array_in >= -Constants.TINY) & (array_in <= 1 + Constants.TINY), axis=0
        )
        solution_mask = np.where(array_to_bool)[0]

        status = _Status.CONVERGED
        if not np.all(array_to_bool):
            status = _Status.OUTBOUND
            total_points = len(array_to_bool)
            outbound_pts = total_points - len(solution_mask)
            logger.warning(
                f"There are {outbound_pts} / {total_points} points out of bounds."
            )

        return status, solution_mask
