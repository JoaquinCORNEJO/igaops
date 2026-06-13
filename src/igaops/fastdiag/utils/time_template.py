from typing import List, Sequence, Optional, Any
from abc import ABC, abstractmethod

from scipy import sparse as sp
import numpy as np

from igaops.common import ParametricDirection, KronProduct
from .template import Template
from .space_precond import SpaceFD


class SpTmFDTemplate(Template, ABC):
    def __init__(self, space_fd: SpaceFD):
        self._space_fd = space_fd
        self._sptm_free_nodes: List[List[int]] = []
        self._advection_time_corrector: Sequence[float] = []

    @property
    def space_fd(self):
        return self._space_fd

    @property
    def nbDoFsPerNode(self):
        return self.space_fd.nbDoFsPerNode

    @property
    def nonzeros_by_dir(self):
        return self.space_fd.nonzeros_by_dir

    @property
    def indices_by_dir(self):
        return self.space_fd.indices_by_dir

    @property
    def tm_nnz(self):
        return self.nonzeros_by_dir.get(ParametricDirection.TAU, 0)

    @property
    def sptm_free_nodes(self):
        return self._sptm_free_nodes

    @property
    def advection_time_corrector(self):
        return self._advection_time_corrector

    def _compute_matrices(self, time_quadrule: Any):
        q = self.rewrite_quadrature([time_quadrule])[0]
        matrices: List[sp.csr_array] = [
            q.weights[0] @ q.basis[0],
            q.weights[1] @ q.basis[1],
        ]
        # We always assume that time is constraint at the begining
        nnz = q.nbctrlpts
        indices = np.arange(1, nnz, dtype=int).tolist()
        self.nonzeros_by_dir[ParametricDirection.TAU] = nnz
        self.indices_by_dir[(0, ParametricDirection.TAU)] = indices
        B = matrices[0].toarray()[np.ix_(indices, indices)]
        A = matrices[1].toarray()[np.ix_(indices, indices)]
        return A, B

    def _propagate_nodes_in_time(self) -> List[List[int]]:
        indx_time = self.indices_by_dir[(0, ParametricDirection.TAU)]
        nnz_time = self.nonzeros_by_dir[ParametricDirection.TAU]
        space_nodes = self.space_fd.space_free_nodes.copy()
        nnz_list = [nnz_time, self.space_fd.sp_nnz]
        free_nodes = [[] for _ in range(self.nbDoFsPerNode)]
        for ii in range(self.nbDoFsPerNode):
            indices_ii_list = [indx_time, space_nodes[ii]]
            global_indices = KronProduct.kron_nonzero_indices(indices_ii_list, nnz_list)
            free_nodes[ii] = global_indices
        return free_nodes

    @abstractmethod
    def compute_time_schurdecomposition(self, time_quadrule: Any):
        """
        Compute the Schur decomposition of the pencil (advection, mass) for the time direction.

        Parameters
        ----------
        time_quadrule : QuadratureRule
            Quadrature rule for the time direction.
        """
        raise NotImplementedError("To implement in children")

    @abstractmethod
    def add_scalar_time_correctors(
        self,
        advection_corrector: Optional[List[float]] = None,
    ):
        """
        Add scalar correctors that takes into account the geometry and material properties.

        It is important to remember that the (vanilla) preconditioner does not consider any information
        about geometry or material properties, then in order to improve the performance of the preconditioner,
        we can add some scalar correctors to the Schur decomposition of the pencil (advection, mass) for the time direction.

        Parameters
        ----------
        advection_corrector : List[float]
            List of scalar correctors for the advection matrix.

        Notes
        -----
        The length of the list should be equal to the number of DoFs per node.
        """
        raise NotImplementedError("To implement in children")

    @abstractmethod
    def apply_spacetime_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        """
        Apply the full space-time fast-diagonalization preconditioner to the input array.

        Parameters
        ----------
        array_in : ndarray
            Input array to which the full space-time preconditioner will be applied.

        Returns
        -------
        ndarray
            Output array after applying preconditioner.
        """
        raise NotImplementedError("To implement in children")

    @abstractmethod
    def apply_spacetime_preconditioner_trans(self, array_in: np.ndarray) -> np.ndarray:
        """
        Apply the full space-time fast-diagonalization preconditioner to the input array.

        Parameters
        ----------
        array_in : ndarray
            Input array to which the full space-time preconditioner will be applied.

        Returns
        -------
        ndarray
            Output array after applying preconditioner.
        """
        raise NotImplementedError("To implement in children")
