from typing import Union, List, Literal, Tuple

from geomdl import BSpline, NURBS
import numpy as np

from igaops.common import ParametricDirection, BoundarySide
from igaops.quadrature import StandardGauss, WeightedQuadrature

from .mixin.projection import ProjectionMixin
from .mixin.transformation import TransformationMixin
from .mixin.reader import Reader
from .mixin.template import SingleSpline, EvaluatedSpline

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


class SinglePatch(ProjectionMixin, TransformationMixin):
    """
    Minimalist patch data structure.

    In addition to SingleSpline properties, it contains
    projection and transformation functionalities.
    """

    @classmethod
    def from_geomdl(
        cls,
        obj: Union[
            BSpline.Curve,
            BSpline.Surface,
            BSpline.Volume,
            NURBS.Curve,
            NURBS.Surface,
            NURBS.Volume,
        ],
        quadclass: Literal["gauss", "weighted"],
        quadtype: Literal["legendre", "lobatto", "1", "2"],
        **quad_args,
    ) -> "SinglePatch":
        """
        Creates a new instance from Geomdl object.

        Extracts essential info from a geomdl object (Curve/Surface/Volume)
        and prepares quadrature + geometric mappings for IGA.

        Parameters
        ----------
        obj : {Curve, Surface, Volume}
            Geomdl object.
        quadclass : {"gauss", "weighted"}
            Quadrature class.
        quadtype : str
            Quadrature type.
        quad_args :  dict
            Extra arguments for quadrature rules.
        """
        instance = cls.__new__(cls)
        ndim = Reader.read_dimensionality(obj)
        degree = Reader.read_degree(obj, ndim)
        knotvector = Reader.read_knotvector(obj, ndim)
        nbctrlpts = [len(kv) - d - 1 for kv, d in zip(knotvector, degree)]
        nbctrlpts = np.asarray(nbctrlpts, dtype=int)
        ctrlpts, nurbs_weights = Reader.read_control_points(obj, nbctrlpts)
        quadrule_list = Reader.set_quadrature_rules(
            degree, knotvector, quadclass, quadtype, **quad_args
        )
        SingleSpline.__init__(
            instance,
            ctrlpts=ctrlpts.copy(),
            nurbs_weights=nurbs_weights.copy(),
            quadrule_list=quadrule_list.copy(),
        )
        return instance

    def generate(self, to_force_trim: bool = False) -> "EvaluatedPatch":
        """
        Initialize the patch by computing geometric transformations at quadrature points.

        Parameters
        ----------
        to_force_trim : bool
            If true, control points, jacobian and its inverse are trimmed. Defaults to true.

        Notes
        -----
        By default Geomdl object always have 3 coordinates (X, Y, Z) but it is not necessarily
        parametrized by 3 variables (e.g. a general curve). So, in general, the jacobian and its inverse
        are rectangular matrices. With ``to_force_trim`` activated, we assume that the object
        has ``ndim`` coordinates for ``ndim`` parameters.
        """
        spline = self.evaluate_spline()
        return EvaluatedPatch.from_spline(spline, to_force_trim=to_force_trim)


class EvaluatedPatch(ProjectionMixin, EvaluatedSpline):
    """
    Fully evaluated patch. Immutable geometric fields.

    In addition to EvaluatedSpline properties,
    it contains projection functionalities.
    """

    @classmethod
    def from_spline(
        cls, spline: EvaluatedSpline, to_force_trim: bool
    ) -> "EvaluatedPatch":
        """
        Creates a new instance from EvaluatedSpline object.

        Parameters
        ----------
        spline : EvaluatedSpline
            Spline object.
        to_force_trim : bool
            If true, control points, jacobian and its inverse are trimmed. Defaults to true.

        Notes
        -----
        By default Geomdl object always have 3 coordinates (X, Y, Z) but it is not necessarily
        parametrized by 3 variables (e.g. a general curve). So, in general, the jacobian and its inverse
        are rectangular matrices. With ``to_force_trim`` activated, we assume that the object
        has ``ndim`` coordinates for ``ndim`` parameters.
        """
        instance = cls.__new__(cls)
        ndim = spline.ndim
        ctrlpts = spline.ctrlpts if not to_force_trim else spline.ctrlpts[:ndim]
        qp_phy = spline.qp_phy if not to_force_trim else spline.qp_phy[:ndim]
        jac = spline.jac if not to_force_trim else spline.jac[:ndim]
        inv_jac = spline.inv_jac if not to_force_trim else spline.inv_jac[:, :ndim]
        EvaluatedSpline.__init__(
            instance,
            ctrlpts=ctrlpts.copy(),
            nurbs_weights=spline.nurbs_weights.copy(),
            quadrule_list=spline.quadrule_list.copy(),
            qp_phy=qp_phy.copy(),
            jac=jac.copy(),
            det_jac=spline.det_jac.copy(),
            inv_jac=inv_jac.copy(),
        )
        return instance

    def get_data_on_boundary(
        self, localization: Tuple[ParametricDirection, BoundarySide]
    ) -> "EvaluatedPatch":
        """
        Computes geometric data on a boundary surface for projection purposes.

        This includes:
        - Physical coordinates of quadrature points on the boundary
        - Jacobian of the transformation at quadrature points
        - NURBS weights at quadrature points (if applicable)
        - Quadrature rules used for integration on the boundary

        Parameters
        ----------
        localization : Tuple[ParametricDirection, BoundarySide]
            Parametric direction and boundary side.
        """

        def get_quadrature_for_boundary(
            obj: SingleSpline,
            localization: Tuple[ParametricDirection, BoundarySide],
        ) -> List[QuadratureRule]:
            quadrules = []
            for ii in range(obj.ndim):
                if ParametricDirection(ii) == localization[0]:
                    continue
                q = StandardGauss(
                    obj.degree[ii], obj.knotvector[ii], quadtype="legendre"
                )
                quadrules.append(q)
            return quadrules

        # Get indices
        indices = self.boundary_manager.select(localization)

        # Get control points and weights on boundary
        ctrlpts = self.ctrlpts[:, indices]
        nurbs_weights = (
            self.nurbs_weights[indices] if self.nurbs_weights.size > 0 else np.array([])
        )

        # Get quadrature on boundary
        quadrules = get_quadrature_for_boundary(self, localization)

        # Mapping: jacobian and determinant
        single_spline = SingleSpline(
            ctrlpts=ctrlpts, nurbs_weights=nurbs_weights, quadrule_list=quadrules
        )
        eval_spline = single_spline.evaluate_spline()
        return EvaluatedPatch.from_spline(eval_spline, to_force_trim=False)
