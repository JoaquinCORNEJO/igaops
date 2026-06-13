from typing import Any, Dict, Tuple, List
from itertools import product

import numpy as np

from igaops.common import ParametricDirection, BoundarySide


def create_connectivity_table(nnz_by_direction: Any) -> np.ndarray:
    """
    Creates a connectivity table for control points based on the number of points in each direction.

    Parameters
    ----------
    nnz_by_direction : ndarray
        Number of control points per direction.

    Returns
    -------
    ndarray
        Connectivity table.
    """
    local = np.asarray(nnz_by_direction, dtype=int)
    indices = np.reshape(np.indices(local.tolist()), (len(local), -1), order="F")
    return np.transpose(indices).astype(int)


class BoundaryManager:

    def __init__(self, nbctrlpts: np.ndarray):
        self.nbctrlpts = nbctrlpts
        self.connectivity_table = create_connectivity_table(nbctrlpts)
        self.boundary_dofs = BoundaryManager._set_boundary_nodes(
            self.nbctrlpts, self.connectivity_table
        )

    def __repr__(self) -> str:
        ndim = len(self.nbctrlpts)
        total = sum(len(v) for v in self.boundary_dofs.values())
        return f"BoundaryManager(ndim={ndim}, nbctrlpts={self.nbctrlpts}, total_boundary_dofs={total})"

    @staticmethod
    def _set_boundary_nodes(
        nbctrlpts: np.ndarray,
        connectivity_table: np.ndarray,
    ) -> Dict[Tuple[ParametricDirection, BoundarySide], List[int]]:
        loc_dir_list = [ParametricDirection(i) for i in range(len(nbctrlpts))]
        boundary_dofs = {}
        for loc_dir, loc_face in product(
            loc_dir_list, [BoundarySide.MIN, BoundarySide.MAX]
        ):
            ld_val = loc_dir.value
            lim = 0 if loc_face == BoundarySide.MIN else nbctrlpts[ld_val] - 1
            nodes = np.where(connectivity_table[:, ld_val] == lim)[0]
            boundary_dofs[(loc_dir, loc_face)] = sorted(nodes.tolist())
        return boundary_dofs

    def select(
        self,
        localization: Tuple[ParametricDirection, BoundarySide],
    ) -> List[int]:
        loc_dir, loc_face = localization

        if not isinstance(loc_dir, ParametricDirection):
            raise TypeError(f"Expected ParametricDirection, got {type(loc_dir)}")
        if not isinstance(loc_face, BoundarySide):
            raise TypeError(f"Expected BoundarySide, got {type(loc_face)}")
        if loc_face == BoundarySide.BOTH:
            raise NotImplementedError("BoundarySide.BOTH is not yet supported")

        return self.boundary_dofs[(loc_dir, loc_face)]
