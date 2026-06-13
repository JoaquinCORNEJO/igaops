from typing import List, Optional, Any
from time import time
import logging

from scipy import linalg as sclin
import numpy as np

from igaops.common import ParametricDirection, Constants
from igaops.operators import MatrixFree
from .time_template import SpTmFDTemplate
from .space_precond import SpaceFD

logger = logging.getLogger("ISOFEM.FASTDIAG")


class SpTimeFD(SpTmFDTemplate):
    def __init__(self, spacefd: SpaceFD):
        super().__init__(spacefd)
        self._schur_adv: np.ndarray = np.array([])
        self._schur_mass: np.ndarray = np.array([])
        self._schur_VSL: np.ndarray = np.array([])
        self._schur_VSR: np.ndarray = np.array([])

    @property
    def schur_adv(self):
        return self._schur_adv

    @property
    def schur_mass(self):
        return self._schur_mass

    @property
    def schur_VSL(self):
        return self._schur_VSL

    @property
    def schur_VSR(self):
        return self._schur_VSR

    def compute_time_schurdecomposition(self, time_quadrule: Any):
        Wt, Mt = self._compute_matrices(time_quadrule)

        S, T, Q, Z = sclin.qz(Wt, Mt, output="complex")
        self._schur_adv = S
        self._schur_mass = T
        self._schur_VSL = Q
        self._schur_VSR = Z

        self._advection_time_corrector = [1.0] * self.nbDoFsPerNode
        self._sptm_free_nodes = self._propagate_nodes_in_time()

    def add_scalar_time_correctors(
        self,
        advection_corrector: Optional[List[float]] = None,
    ):

        def is_verified(entry):
            return isinstance(entry, (list, tuple)) and len(entry) == self.nbDoFsPerNode

        if advection_corrector is not None and is_verified(advection_corrector):
            self._advection_time_corrector = advection_corrector

    def apply_spacetime_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        start = time()
        tail_shape = array_in.shape[1:]
        array_in = np.reshape(array_in, (self.nbDoFsPerNode, -1, *tail_shape))
        array_out = np.zeros_like(array_in)
        adv_corr = self.advection_time_corrector
        nnz_time = len(self.indices_by_dir[(0, ParametricDirection.TAU)])

        for ii in range(self.nbDoFsPerNode):
            array_to_apply = array_in[ii][self.sptm_free_nodes[ii]]
            if np.linalg.norm(array_to_apply) == 0.0:
                continue

            array1 = MatrixFree.apply(
                self.space_fd.eigenvec_by_dir_space[ii] + [self.schur_VSL.conj()],
                array_to_apply,
                is_transpose=True,
            )
            # NOTE: Here, the order "F" has the meaning of slicing in time
            # so maybe it's not worth spending time in changing it to "C"
            array1_reshape = np.reshape(array1, (-1, nnz_time, *tail_shape), order="F")
            array2_reshape = np.zeros_like(array1_reshape)

            eigenvalues = self.space_fd.space_eigenvalues[ii].copy()
            if np.any(eigenvalues <= Constants.TINY):
                # Apply regularization
                logger.warning("Apply regularization")
                eigenvalues += Constants.TINY * np.max(eigenvalues)

            # FIXME: Could we apply multi-threading here since we have
            # to solve many independent triangular systems?
            for idx, row in enumerate(array1_reshape):
                mat = adv_corr[ii] * self.schur_adv + eigenvalues[idx] * self.schur_mass
                array2_reshape[idx] = sclin.solve_triangular(mat, row, lower=False)

            tmp = np.real(
                MatrixFree.apply(
                    self.space_fd.eigenvec_by_dir_space[ii] + [self.schur_VSR],
                    np.reshape(array2_reshape, (-1, *tail_shape), order="F"),
                    is_transpose=False,
                )
            )

            array_out[ii][self.sptm_free_nodes[ii]] = tmp

        logger.debug(
            f"Single patch fast-diagonalization in {time() - start:.2e} seconds"
        )
        return np.reshape(array_out, (-1, *tail_shape))

    def apply_spacetime_preconditioner_trans(self, array_in: np.ndarray) -> np.ndarray:
        start = time()
        tail_shape = array_in.shape[1:]
        array_in = np.reshape(array_in, (self.nbDoFsPerNode, -1, *tail_shape))
        array_out = np.zeros_like(array_in)
        adv_corr = self.advection_time_corrector
        nnz_time = len(self.indices_by_dir[(0, ParametricDirection.TAU)])

        for ii in range(self.nbDoFsPerNode):
            array_to_apply = array_in[ii][self.sptm_free_nodes[ii]]
            if np.linalg.norm(array_to_apply) == 0.0:
                continue

            array1 = MatrixFree.apply(
                self.space_fd.eigenvec_by_dir_space[ii] + [self.schur_VSR],
                array_to_apply,
                is_transpose=True,
            )

            # NOTE: Here, the order "F" has the meaning of slicing in time
            # so maybe it's not worth spending time in changing it to "C"
            array1_reshape = np.reshape(array1, (-1, nnz_time, *tail_shape), order="F")
            array2_reshape = np.zeros_like(array1_reshape)

            eigenvalues = self.space_fd.space_eigenvalues[ii].copy()
            if np.any(eigenvalues <= Constants.TINY):
                # Apply regularization
                logger.warning("Apply regularization")
                eigenvalues += Constants.TINY * np.max(eigenvalues)

            # FIXME: Could we apply multi-threading here since we have
            # to solve many independent triangular systems?
            for idx, row in enumerate(array1_reshape):
                mat = (
                    adv_corr[ii] * self.schur_adv.T
                    + eigenvalues[idx] * self.schur_mass.T
                )
                array2_reshape[idx] = sclin.solve_triangular(mat, row, lower=True)

            tmp = np.real(
                MatrixFree.apply(
                    self.space_fd.eigenvec_by_dir_space[ii] + [self.schur_VSL.conj()],
                    np.reshape(array2_reshape, (-1, *tail_shape), order="F"),
                    is_transpose=False,
                )
            )

            array_out[ii][self.sptm_free_nodes[ii]] = tmp

        logger.debug(
            f"Single patch fast-diagonalization in {time() - start:.2e} seconds"
        )
        return np.reshape(array_out, (-1, *tail_shape))
