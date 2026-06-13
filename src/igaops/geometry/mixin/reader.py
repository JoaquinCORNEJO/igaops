from typing import Union, List, Literal, Tuple, Any
import numpy as np

from igaops.common import validate_entry
from igaops.quadrature import StandardGauss, WeightedQuadrature

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


class Reader:
    """
    Read and save information from object created by Geomdl.
    """

    @staticmethod
    def ensure_3_rows(arr: np.ndarray):
        """
        Converts a matrix of size nxm to 3xm.

        Parameters
        ----------
        arr : ndarray
            Matrix to convert.

        Raises
        ------
        ValueError
            If array is not 2-dimensional or if n > 3.
        """
        arr = np.asarray(arr)

        # Check that the input is 2D
        if arr.ndim != 2:
            raise ValueError("Input array must be 2-dimensional.")

        nr, nc = arr.shape

        if nr > 3:
            raise ValueError(f"Input array has {nr} rows, but at most 3 are allowed.")

        # Create output array filled with zeros
        out = np.zeros((3, nc), dtype=arr.dtype)

        # Copy existing columns
        out[:nr, :] = arr

        return out

    @staticmethod
    def read_dimensionality(obj: Any) -> int:
        """
        Reads the dimensionality of the spline in its parametric space.

        Parameters
        ----------
        obj : any
            Geomdl object.
        """
        is_curve = (
            hasattr(obj, "degree")
            and hasattr(obj, "knotvector")
            and not hasattr(obj, "knotvector_v")
        )

        is_surface = (
            hasattr(obj, "degree_u")
            and hasattr(obj, "degree_v")
            and not hasattr(obj, "degree_w")
        )

        is_volume = (
            hasattr(obj, "degree_u")
            and hasattr(obj, "degree_v")
            and hasattr(obj, "degree_w")
        )

        if is_volume:
            ndim = 3
        elif is_surface:
            ndim = 2
        elif is_curve:
            ndim = 1
        else:
            raise TypeError("Geometry is not a supported geomdl Curve/Surface/Volume.")
        return ndim

    @staticmethod
    def read_degree(obj: Any, ndim: int) -> np.ndarray:
        """
        Reads the polynomial degrees in each of its parametric dimensions.

        Parameters
        ----------
        obj : any
            Geomdl object.
        ndim : int
            Dimensionality of its parametric space.
        """
        degree = []
        if ndim == 1:
            degree.append(obj.degree)
        else:
            degree.append(obj.degree_u)
            degree.append(obj.degree_v)
            if ndim == 3:
                degree.append(obj.degree_w)
        return np.asarray(degree, dtype=int)

    @staticmethod
    def read_knotvector(obj: Any, ndim: int) -> List[np.ndarray]:
        """
        Reads the knotvectors in each of its parametric dimensions.

        Parameters
        ----------
        obj : any
            Geomdl object.
        ndim : int
            Dimensionality of its parametric space.
        """
        kv: List[np.ndarray] = []
        if ndim == 1:
            kv.append(np.asarray(obj.knotvector, dtype=float))
        else:
            kv.append(np.asarray(obj.knotvector_u, dtype=float))
            kv.append(np.asarray(obj.knotvector_v, dtype=float))
            if ndim == 3:
                kv.append(np.asarray(obj.knotvector_w, dtype=float))
        return kv

    @staticmethod
    def read_control_points(
        obj: Any, nbctrlpts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads the control points. By default it is in 3D coordinates.

        Parameters
        ----------
        obj : any
            Geomdl object.
        nbctrlpts : int
            Number of control points on each dimension.
        """
        nr = 3
        nnz = np.ones(nr, dtype=int)
        nnz[: len(nbctrlpts)] = nbctrlpts

        ctrlpts = Reader.ensure_3_rows(np.asarray(obj.ctrlpts).T)
        nurbs_weights = np.asarray(obj.weights or np.array([]))

        old_grid = np.reshape(ctrlpts, (nr, nnz[2], nnz[0], nnz[1]))
        new_grid = np.swapaxes(old_grid, 2, 3)
        ctrlpts = np.reshape(new_grid, (nr, -1))

        if nurbs_weights.size > 0:
            old_grid = np.reshape(nurbs_weights, (nnz[2], nnz[0], nnz[1]))
            new_grid = np.swapaxes(old_grid, 1, 2)
            nurbs_weights = np.ravel(new_grid)

        return ctrlpts, nurbs_weights

    @staticmethod
    def set_quadrature_rules(
        degree: np.ndarray,
        knotvector: List[np.ndarray],
        quadclass: Literal["gauss", "weighted"],
        quadtype: Any,
        **quad_args,
    ) -> List[QuadratureRule]:
        """
        Set the quadrature rules for each dimension.

        Parameters
        ----------
        degree : list[int]
            Sequence of degree on each dimension.
        knotvector : list[ndarray]
            Sequence of knotvectors on each dimension.
        quadclass : {"gauss", "weighted"}
            Quadrature class.
        quadtype : str
            Quadrature type.
        quad_args : dict
            Extra arguments for quadrature rules.
        """

        validate_entry(quadclass, ["gauss", "weighted"])

        # Default values
        quadrule_cls = StandardGauss if quadclass == "gauss" else WeightedQuadrature
        quadtype_ref = ["legendre", "lobatto"] if quadclass == "gauss" else ["1", "2"]

        validate_entry(quadtype, quadtype_ref)

        quadrule_list: List[QuadratureRule] = []
        for d, kv in zip(degree, knotvector):
            q = quadrule_cls(d, kv, quadtype, **quad_args)
            quadrule_list.append(q)
        return quadrule_list
