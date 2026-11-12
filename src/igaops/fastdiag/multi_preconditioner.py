from typing import Sequence, List, Literal, Any
from time import time
import inspect
import logging

import numpy as np

from .preconditioner import SingleFastDiagonalization as SingleFD

logger = logging.getLogger(__name__)


class MultiFastDiagonalization:
    """
    Parameters
    ----------
    list_preconditioner : List[SingleFD]
        List of single-patch preconditioners.
    """

    def __init__(self, list_preconditioner: Sequence[SingleFD]):
        if len(list_preconditioner) < 1:
            raise RuntimeError("Empty list of preconditioners")

        if not all(isinstance(m, SingleFD) for m in list_preconditioner):
            raise RuntimeError("All elements should be preconditioners")

        self._list_fd = list_preconditioner
        self._preconditioner_type: Any = list_preconditioner[0].preconditioner_type
        self._slices = self._set_slices(list_preconditioner)

        for fd in list_preconditioner:
            if fd.preconditioner_type != self.preconditioner_type:
                raise RuntimeError("Preconditioners should be of the same type")

    @property
    def preconditioner_type(self) -> Literal["spatial", "spacetime"]:
        return self._preconditioner_type

    def _set_slices(self, list_fd: Sequence[SingleFD]) -> List[slice]:
        array: List[int] = [0]
        for fd in list_fd:
            sp_fd = fd.space_preconditioner
            sp_nnz = sp_fd.nbDoFsPerNode * sp_fd.sp_nnz
            tm_nnz = 1
            if self.preconditioner_type == "spacetime":
                tm_nnz = fd.sptm_preconditioner.tm_nnz
            array.append(int(sp_nnz * tm_nnz))
        cumsum_sizes = np.cumsum(array).astype(int)
        return [
            slice(cumsum_sizes[i], cumsum_sizes[i + 1])
            for i in range(len(cumsum_sizes) - 1)
        ]

    def _apply_func(self, array_in: np.ndarray, func_name: str) -> np.ndarray:
        array_out = np.zeros_like(array_in)
        for fd, slc in zip(self._list_fd, self._slices):
            array_out[slc] = fd.__getattribute__(func_name)(array_in[slc])
        return array_out

    def apply(self, array_in: np.ndarray) -> np.ndarray:
        """
        Apply the fast-diagonalization preconditioner to the input array.

        Parameters
        ----------
        array_in : ndarray
            Input array to which the preconditioner will be applied.

        Returns
        -------
        ndarray
            Output array after applying preconditioner.
        """
        fname = str(inspect.currentframe().f_code.co_name)
        start = time()
        array_out = self._apply_func(array_in, fname)
        logger.debug(
            f"Multi-patch fast-diagonalization in {time() - start:.2e} seconds"
        )
        return array_out

    def apply_transpose(self, array_in: np.ndarray) -> np.ndarray:
        """
        Apply the transpose of the fast-diagonalization preconditioner to the input array.

        Parameters
        ----------
        array_in : ndarray
            Input array to which the preconditioner will be applied.

        Returns
        -------
        ndarray
            Output array after applying preconditioner.
        """
        fname = str(inspect.currentframe().f_code.co_name)
        start = time()
        array_out = self._apply_func(array_in, fname)
        logger.debug(
            f"Multi-patch fast-diagonalization in {time() - start:.2e} seconds"
        )
        return array_out
