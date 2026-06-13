from typing import Union, List, Tuple, Optional

from scipy import sparse as sp
import numpy as np

from igaops.common import Constants
from igaops.quadrature import StandardGauss, WeightedQuadrature
from igaops.operators import NurbsOperations, BsplineOperations

from .boundary_manager import BoundaryManager
from .operations import Operations

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


class SingleSpline:
    """
    Minimalist datastructure for a spline.

    It contains:
        - Control points
        - Nurbs weights (zero array if Bspline)
        - Quadrature rules (in each direction)
    """

    def __init__(
        self,
        ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray,
        quadrule_list: List[QuadratureRule],
    ):
        self.ctrlpts = ctrlpts
        self.nurbs_weights = nurbs_weights
        self.quadrule_list = quadrule_list
        self._boundary_manager: Optional[BoundaryManager] = None

    @property
    def ndim(self):
        "Dimensionality of parametric space"
        return len(self.quadrule_list)

    @property
    def degree(self):
        "List of degrees"
        return np.asarray([self.quadrule_list[i].degree for i in range(self.ndim)])

    @property
    def knotvector(self):
        "List of knotvectors"
        return [self.quadrule_list[i].knotvector for i in range(self.ndim)]

    @property
    def nbctrlpts(self):
        "Number of control points"
        nb = [self.quadrule_list[i].nbctrlpts for i in range(self.ndim)]
        return np.asarray(nb, dtype=int)

    @property
    def nbquadpts(self):
        "Number of quadrature points"
        nb = [self.quadrule_list[i].nbquadpts for i in range(self.ndim)]
        return np.asarray(nb, dtype=int)

    @property
    def nbctrlpts_total(self) -> int:
        "Total number of control points (product over all parametric directions)."
        return np.prod(self.nbctrlpts, dtype=int)

    @property
    def nbquadpts_total(self) -> int:
        "Total number of quadrature points (product over all parametric directions)."
        return np.prod(self.nbquadpts, dtype=int)

    @property
    def operator_engine(self):
        "Bspline or NURBS class"
        return NurbsOperations if self.nurbs_weights.size > 0 else BsplineOperations

    @property
    def boundary_manager(self):
        if self._boundary_manager is None:
            self._boundary_manager = BoundaryManager(self.nbctrlpts)
        return self._boundary_manager

    def evaluate_spline(self) -> "EvaluatedSpline":
        """
        Compute |J|, J, J^{-1} and physical coordinates at quadrature points.

        Returns
        -------
        Spline
            Object to save geometrical information (det_jac, jac, inv_jac).

        Raises
        ------
        ValueError
            If any quadrature point has non-positive Jacobian determinant.
        """
        basis_list = [q.basis for q in self.quadrule_list]

        jac = self.operator_engine.eval_jacobien(
            basis_list, self.ctrlpts, nurbs_weights=self.nurbs_weights
        )  # shape: (spatial_dim, param_dim, ...)

        det_jac, inv_jac = Operations.inverse_rectangular_matrix(jac)

        knots_phy = self.operator_engine.eval_interpolation(
            basis_list, self.ctrlpts, nurbs_weights=self.nurbs_weights
        )

        if not np.all(det_jac > 0.0):
            nb_bad = int(np.sum(det_jac <= 0.0))
            frac = 100.0 * nb_bad / det_jac.size
            msg = (
                f"Potential geometry issue: {frac:.2e}% quadrature points have detJ <= 0 "
                f"({nb_bad}/{det_jac.size})."
            )
            raise ValueError(msg)

        spline = EvaluatedSpline(
            ctrlpts=self.ctrlpts.copy(),
            nurbs_weights=self.nurbs_weights.copy(),
            qp_phy=knots_phy.copy(),
            jac=jac.copy(),
            det_jac=det_jac.copy(),
            inv_jac=inv_jac.copy(),
            quadrule_list=self.quadrule_list.copy(),
        )

        return spline

    def __repr__(self) -> str:
        message = f""""
            GEOMETRY:
            dimensionality: {self.ndim}
            polynomial degrees: {self.degree.tolist()}
            nb of control points: {self.nbctrlpts.tolist()}
            total nb of c.p. : {self.nbctrlpts_total}
            nb of quadrature points: {self.nbquadpts.tolist()}
            total nb of q.p. : {self.nbquadpts_total}
        """
        return message


class EvaluatedSpline(SingleSpline):
    """
    Fully evaluated datastructure for a spline. Immutable geometric fields.

    It contains:
        - Control points
        - Nurbs weights (zero array if Bspline)
        - Quadrature rules (in each direction)
        - Quadrature points in physical space
        - Jacobian of transformation
        - Determinant of Jacobian
        - Inverse of Jacobian
    """

    def __init__(
        self,
        ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray,
        quadrule_list: List[QuadratureRule],
        qp_phy: np.ndarray,
        jac: np.ndarray,
        det_jac: np.ndarray,
        inv_jac: np.ndarray,
    ):
        SingleSpline.__init__(
            self,
            ctrlpts=ctrlpts,
            nurbs_weights=nurbs_weights,
            quadrule_list=quadrule_list,
        )
        self.qp_phy = qp_phy
        self.jac = jac
        self.det_jac = det_jac
        self.inv_jac = inv_jac

    @property
    def box_corners(self) -> np.ndarray:
        """
        Return the coordinates of the corners of the bounding box of the patch.
        """
        min_coords = np.min(self.ctrlpts[: self.ndim, :], axis=1)
        max_coords = np.max(self.ctrlpts[: self.ndim, :], axis=1)
        if self.ndim == 1:
            corners = np.array([[min_coords[0]], [max_coords[0]]])
        elif self.ndim == 2:
            corners = np.array(
                [
                    [min_coords[0], min_coords[1]],
                    [max_coords[0], min_coords[1]],
                    [min_coords[0], max_coords[1]],
                    [max_coords[0], max_coords[1]],
                ]
            ).T
        elif self.ndim == 3:
            corners = np.array(
                [
                    [min_coords[0], min_coords[1], min_coords[2]],
                    [max_coords[0], min_coords[1], min_coords[2]],
                    [min_coords[0], max_coords[1], min_coords[2]],
                    [max_coords[0], max_coords[1], min_coords[2]],
                    [min_coords[0], min_coords[1], max_coords[2]],
                    [max_coords[0], min_coords[1], max_coords[2]],
                    [min_coords[0], max_coords[1], max_coords[2]],
                    [max_coords[0], max_coords[1], max_coords[2]],
                ]
            ).T
        else:
            raise ValueError("Invalid number of dimensions.")
        return corners

    @property
    def characteristic_length(self) -> float:
        """
        Return a characteristic length of the patch, defined as the maximum distance in the box corners.
        """
        corners = self.box_corners
        max_distance = 0.0
        for i in range(corners.shape[1]):
            for j in range(i + 1, corners.shape[1]):
                distance = np.linalg.norm(corners[:, i] - corners[:, j])
                if distance > max_distance:
                    max_distance = float(distance)
        return max_distance

    def compute_global_mesh_parameter(self) -> Tuple[float, float]:
        """
        Return the maximum unique-knot spacing over all parametric directions.
        """
        max_distance = []
        for i in range(self.ndim):
            q = self.quadrule_list[i]
            max_distance.append(q.max_h_size)
        max_h = float(np.max(max_distance)) if max_distance else 0.0
        min_p = float(np.min(self.degree))
        return max_h, min_p

    def project_skeleton_to_physical_space(self, n: int = 11):
        """
        Computes the skeleton of a spline projected in the physical space.

        From a parametric space point of view:
        - For 1D spline, the skeleton are just the endpoints [0, 1]
        - For 2D spline, the skeleton is the set [0, 1] x {0, 1} U {0, 1} x [0, 1]
        - For 3D spline, it is not yet implemented.
        """
        lowb = Constants.TINY
        higb = 1 - lowb

        if self.ndim == 1:
            quadpts = [np.array([lowb, higb])]
        elif self.ndim == 2:
            flnsp = np.linspace(lowb, higb, n)
            blnsp = flnsp[::-1]
            pairs = []
            for i in range(n):
                pairs.append([flnsp[i], lowb])
            for i in range(1, n):
                pairs.append([higb, flnsp[i]])
            for i in range(1, n):
                pairs.append([blnsp[i], higb])
            for i in range(1, n):
                pairs.append([lowb, blnsp[i]])
            quadpts = np.asarray(pairs).T.tolist()
        else:
            raise ValueError("Expected curve or surface.")

        basis_list: List[List[sp.csr_array]] = []
        for i, q in enumerate(self.quadrule_list):
            b = q.eval_basis(quadpts[i])
            basis_list.append(b)

        return self.operator_engine.eval_interpolation(
            basis_list, self.ctrlpts, nurbs_weights=self.nurbs_weights, along_axis=True
        )
