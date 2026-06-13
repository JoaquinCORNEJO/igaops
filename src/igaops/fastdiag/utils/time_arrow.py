from typing import Sequence, List, Optional, Any
from time import time
import logging

from scipy import linalg as sclin
import numpy as np

from igaops.common import Constants
from igaops.operators import MatrixFree
from .operations import solve_special_arrowhead, compute_special_schur
from .time_template import SpTmFDTemplate
from .space_precond import SpaceFD

logger = logging.getLogger(__name__)


class SpTimeFD(SpTmFDTemplate):
    def __init__(self, space_fd: SpaceFD):
        super().__init__(space_fd)
        self._eigenvec_time: np.ndarray = np.array([])
        self._eigenblocks_time: Sequence[np.ndarray] = []
        self._blocks_time = None
        self._blocks_time_T = None

    @property
    def eigenvec_time(self):
        return self._eigenvec_time

    @property
    def eigenblocks_time(self):
        return self._eigenblocks_time

    @property
    def blocks_time(self):
        if self._blocks_time is None:
            self._blocks_time = self._compute_blocks(is_transpose=False)
        return self._blocks_time

    @property
    def blocks_time_transpose(self):
        if self._blocks_time_T is None:
            self._blocks_time_T = self._compute_blocks(is_transpose=True)
        return self._blocks_time_T

    def compute_time_schurdecomposition(self, time_quadrule: Any):
        Wt, Mt = self._compute_matrices(time_quadrule)

        # Extract blocks
        Wp = Wt[:-1, :-1]
        Mp = Mt[:-1, :-1]
        m = Mt[:-1, -1]
        w = np.atleast_2d(Wt[:-1, -1])

        # Compute eigen decomposition
        Dtp, Utp = sclin.eig(Wp, Mp)
        for i in range(Utp.shape[1]):
            u_i = Utp[:, i].copy()
            Utp[:, i] /= np.sqrt(np.abs(u_i.conj().T @ Mp @ u_i))

        # Compute [v, 1] and normalized it -> [k rho]
        v = np.linalg.solve(Mp, -m)
        v_1 = np.hstack([v, [1]])
        k_rho = v_1 / np.sqrt(v_1 @ (Mt @ v_1))
        k = np.atleast_2d(k_rho[:-1])

        # Assemble Ut: Ut^H @ Mt @ Ut = It
        Ut = np.block([[Utp, k.T], [np.zeros_like(k), k_rho[-1]]])

        # Compute blocks of Dt
        g = np.atleast_2d(Utp.conj().T @ (np.block([Wp, w.T]) @ k_rho))
        sigma = k_rho @ (Wt @ k_rho)

        # Assemble Dt: Ut^H @ Wt @ Ut = Dt
        Dt = np.block([[np.diag(Dtp), g.T], [-g.conj(), sigma]])

        # We only need to save Ut and the blocks of Dt (not necessary Dt)
        self._eigenvec_time = Ut
        self._eigenblocks_time = [np.diag(Dt), g.ravel()]

        self._advection_time_corrector = [1.0] * self.nbDoFsPerNode
        self._sptm_free_nodes = self._propagate_nodes_in_time()

    def _compute_blocks(self, is_transpose=False):
        adv_corr = self.advection_time_corrector
        space_eigvals = self.space_fd.space_eigenvalues
        Dt, g = self.eigenblocks_time
        g = -g.conj() if is_transpose else g
        bfac_list: List[np.ndarray] = []
        H_list: List[np.ndarray] = []
        schur_list: List[np.ndarray] = []
        for ii in range(self.nbDoFsPerNode):
            eigenvalues = space_eigvals[ii].copy()
            if np.any(eigenvalues <= Constants.TINY):
                # Apply regularization
                logger.warning("Apply regularization")
                eigenvalues += Constants.TINY * np.max(eigenvalues)
            bfac = adv_corr[ii] * g
            H = np.add.outer(adv_corr[ii] * Dt, eigenvalues)
            S = compute_special_schur(H, bfac)
            bfac_list.append(bfac)
            H_list.append(H)
            schur_list.append(S)
        return H_list, bfac_list, schur_list

    def add_scalar_time_correctors(
        self,
        advection_corrector: Optional[List[float]] = None,
    ):

        def is_verified(entry):
            return isinstance(entry, (list, tuple)) and len(entry) == self.nbDoFsPerNode

        if advection_corrector is not None and is_verified(advection_corrector):
            self._advection_time_corrector = advection_corrector
            self._blocks_time = None

    def apply_spacetime_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        start = time()
        tail_shape = array_in.shape[1:]
        array_in = np.reshape(array_in, (self.nbDoFsPerNode, -1, *tail_shape))
        array_out = np.zeros_like(array_in)
        H, bfac, S = self.blocks_time

        for ii in range(self.nbDoFsPerNode):
            array_to_apply = array_in[ii][self.sptm_free_nodes[ii]]
            if np.linalg.norm(array_to_apply) == 0.0:
                continue

            array1 = MatrixFree.apply(
                self.space_fd.eigenvec_by_dir_space[ii] + [self.eigenvec_time.conj()],
                array_to_apply,
                is_transpose=True,
            )

            array2 = solve_special_arrowhead(
                H=H[ii], bfac=bfac[ii], rhs=array1, schur=S[ii]
            )

            tmp = np.real(
                MatrixFree.apply(
                    self.space_fd.eigenvec_by_dir_space[ii] + [self.eigenvec_time],
                    array2,
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
        H, bfac, S = self.blocks_time_transpose

        for ii in range(self.nbDoFsPerNode):
            array_to_apply = array_in[ii][self.sptm_free_nodes[ii]]
            if np.linalg.norm(array_to_apply) == 0.0:
                continue

            array1 = MatrixFree.apply(
                self.space_fd.eigenvec_by_dir_space[ii] + [self.eigenvec_time],
                array_to_apply,
                is_transpose=True,
            )

            array2 = solve_special_arrowhead(
                H=H[ii], bfac=bfac[ii], rhs=array1, schur=S[ii]
            )

            tmp = np.real(
                MatrixFree.apply(
                    self.space_fd.eigenvec_by_dir_space[ii]
                    + [self.eigenvec_time.conj()],
                    array2,
                    is_transpose=False,
                )
            )

            array_out[ii][self.sptm_free_nodes[ii]] = tmp

        logger.debug(
            f"Single patch fast-diagonalization in {time() - start:.2e} seconds"
        )
        return np.reshape(array_out, (-1, *tail_shape))
