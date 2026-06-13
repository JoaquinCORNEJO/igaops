from typing import Optional, Literal
import logging

import numpy as np

from igaops.common import Constants, validate_entry
from .template import SingleSpline

logger = logging.getLogger(__name__)


class TransformationMixin(SingleSpline):
    def scale(self, factor: float):
        """
        Zoom in or zoom out.

        Multiply the values of the control points by a given factor in order to zoom in or zoom out.

        Parameters
        ----------
        factor : float
            Scaling fector to zoom in or zoom out.
        """
        if np.abs(factor) < Constants.TINY:
            raise ValueError("Scaling factor is too small")
        logger.info("Scaling single patch")
        for i in range(self.ctrlpts.shape[0]):
            self.ctrlpts[i] *= np.abs(factor)
        return self

    def translate(self, vector: np.ndarray):
        """
        Translate control points by a vector in-place.

        Rigid displacement of the control points.

        Parameters
        ----------
        vector : ndarray
            Translation vector.
        """
        vector = np.asarray(vector, dtype=float)
        if vector.ndim != 1 and len(vector) != 3:
            raise ValueError("Array should be 1D of size 3")
        logger.info("Translating single patch")
        for i in range(min(self.ctrlpts.shape[0], vector.size)):
            self.ctrlpts[i] += vector[i]
        return self

    def reflect(self, plane: Literal["xy", "xz", "yz"]):
        """
        Reflect across planes: 'xy', 'xz', 'yz' (order-insensitive).

        Parameters
        ----------
        plane : {"xy", "xz", "yz"}
            Reflect plane.
        """
        plane = validate_entry(plane, ["xy", "xz", "yz"])
        logger.info("Reflecting single patch")
        axis = (set("xyz") - set(plane)).pop()  # the normal axis
        idx = ["x", "y", "z"].index(axis)
        self.ctrlpts[idx] *= -1.0
        return self

    def rotate(
        self,
        axis: Literal["x", "y", "z"],
        angle: float,
        point: Optional[np.ndarray] = None,
    ):
        """
        Rotate the control points around a given axis ('x', 'y', or 'z') by a given angle (in radians),
        around a given point (default is the origin).

        Parameters
        ----------
        axis : {"x", "y", "z"}
            Rotation axis.
        angle : float
            Rotation angle in radians.
        point : ndarray, optional
            Rotation point. Defaults to origin (0, 0, 0).
        """
        axis = validate_entry(axis, ["x", "y", "z"])
        angle = float(angle)
        if point is None:
            point = np.zeros(3)
        point = np.asarray(point, dtype=float)
        if point.ndim != 1 and len(point) != 3:
            raise ValueError("Array should be 1D of size 3")

        logger.info("Rotating single patch")

        # Move to rotation center
        self.ctrlpts -= point[:, np.newaxis]

        # Build rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.eye(3)

        if axis == "x":
            R[1, 1] = c
            R[1, 2] = -s
            R[2, 1] = s
            R[2, 2] = c
        if axis == "y":
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c
        if axis == "z":
            R[0, 0] = c
            R[0, 1] = -s
            R[1, 0] = s
            R[1, 1] = c

        # Apply rotation
        self.ctrlpts = R @ self.ctrlpts

        # Move back
        self.ctrlpts += point[:, np.newaxis]
        return self
