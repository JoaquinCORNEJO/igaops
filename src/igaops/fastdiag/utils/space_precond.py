from typing import List, Sequence, Optional, Dict, Tuple, Any
from time import time
import logging

from scipy import linalg as sclin, sparse as sp
import numpy as np

from igaops.common import (
    Constants,
    ParametricDirection,
    KronProduct,
    combine_arrays,
)
from igaops.operators import MatrixFree
from .template import Template

logger = logging.getLogger(__name__)


class SpaceFD(Template):
    def __init__(self):
        # Imperative variables
        self._nonzeros_by_dir: Dict[ParametricDirection, int] = {}
        self._indices_by_dir: Dict[Tuple[int, ParametricDirection], List[int]] = {}
        self._nbDoFsPerNode: int = 0

        self._space_dirs: List[ParametricDirection] = []
        self._space_free_nodes: List[List[int]] = []
        self._space_table_dirichlet: np.ndarray = np.array([])

        # NOTE: FD was concieve as preconditioner for stiffness
        # Then, we initialized it with (0.0, 1.0)
        self._space_scalar_coefs: Tuple[float, float] = (0.0, 1.0)
        self._space_eigenvalues: Optional[List[np.ndarray]] = None
        self._eigenvec_by_dir_space: List[List[np.ndarray]] = []
        self._eigenval_by_dir_space: List[List[np.ndarray]] = []
        self._mass_space_corrector: Sequence[float] = []
        self._stiff_space_corrector: Sequence[np.ndarray] = []

    @property
    def sp_nnz(self):
        return np.prod(
            [
                val
                for key, val in self.nonzeros_by_dir.items()
                if key != ParametricDirection.TAU
            ]
        )

    @property
    def nonzeros_by_dir(self):
        return self._nonzeros_by_dir

    @property
    def indices_by_dir(self):
        return self._indices_by_dir

    @property
    def nbDoFsPerNode(self):
        return self._nbDoFsPerNode

    @property
    def space_dirs(self):
        return self._space_dirs

    @property
    def space_free_nodes(self):
        return self._space_free_nodes

    @property
    def space_table_dirichlet(self):
        return self._space_table_dirichlet

    @property
    def space_scalar_coefs(self):
        return self._space_scalar_coefs

    @property
    def space_eigenvalues(self):
        if self._space_eigenvalues is not None:
            return self._space_eigenvalues
        space_eigenvalues = self.update_space_eigenvalues(
            scalar_coefs=self.space_scalar_coefs
        )
        self._space_eigenvalues = space_eigenvalues
        return space_eigenvalues

    @property
    def eigenvec_by_dir_space(self):
        return self._eigenvec_by_dir_space

    @property
    def eigenval_by_dir_space(self):
        return self._eigenval_by_dir_space

    @property
    def mass_space_corrector(self):
        return self._mass_space_corrector

    @property
    def stiff_space_corrector(self):
        return self._stiff_space_corrector

    def _propagate_nodes_in_space(self):
        indices_by_dir = self.indices_by_dir
        nonzeros_by_dir = self.nonzeros_by_dir
        space_free_nodes = [[] for _ in range(self.nbDoFsPerNode)]

        if not (len(indices_by_dir) > 0 and len(nonzeros_by_dir) > 0):
            return space_free_nodes

        nnz_list = [nonzeros_by_dir[dirs] for dirs in self.space_dirs]
        nnz_list.reverse()
        for ii in range(self.nbDoFsPerNode):
            # NOTE: we inverse the list in order to coincide with (M_n x ... x M_1)
            # See matrix-free algorithms
            indices_list = [
                indices_by_dir.get((ii, dirs), []) for dirs in self.space_dirs[::-1]
            ]
            global_indices = KronProduct.kron_nonzero_indices(indices_list, nnz_list)
            space_free_nodes[ii] = global_indices
        return space_free_nodes

    def compute_space_eigendecomposition(
        self,
        space_quadrule_list: Sequence[Any],
        space_table_dirichlet: np.ndarray,
    ):
        """
        Compute the eigendecomposition of the pencils (stiffness, mass) for each direction in space and for each DoFs per node.

        Parameters
        ----------
        space_quadrule_list : List[QuadratureRule]
            List of quadrature rules for each spatial direction.
        space_table_dirichlet : ndarray
            A boolean array indicating the Dirichlet boundary conditions for each DoFs per node and direction.
        """
        quadrule_list = self.rewrite_quadrature(space_quadrule_list)
        mass_list: List[sp.csr_array] = [
            q.weights[0] @ q.basis[0] for q in quadrule_list
        ]
        stiff_list: List[sp.csr_array] = [
            q.weights[-1] @ q.basis[-1] for q in quadrule_list
        ]

        nbDoFsPerNode, nbdirs, _ = np.shape(space_table_dirichlet)
        nbdirs = np.min([nbdirs, len(quadrule_list)])
        space_dirs = [ParametricDirection(i) for i in range(nbdirs)]

        nnz_by_dir, indx_by_dir = {}, {}
        mass_corrector, stif_corrector = [], []
        all_eigvecs_by_dir, all_eigvals_by_dir = [], []
        for ii in range(nbDoFsPerNode):
            eigvecs_by_dir, eigvals_by_dir = [], []
            mass_corrector.append(1.0)
            stif_corrector.append(np.ones(nbdirs))
            for jj, sp_dir in enumerate(space_dirs):
                B: np.ndarray = mass_list[jj].toarray()
                A: np.ndarray = stiff_list[jj].toarray()
                nnz = B.shape[0]
                inf_index, sup_index = 0, nnz
                if space_table_dirichlet[ii, jj, 0]:
                    inf_index += 1
                if space_table_dirichlet[ii, jj, 1]:
                    sup_index -= 1
                indices = np.arange(inf_index, sup_index, dtype=int).tolist()
                nnz_by_dir[sp_dir] = nnz
                indx_by_dir[(ii, sp_dir)] = indices
                if nnz == len(indices):
                    # NOTE: this avoids division by zero
                    A += Constants.SMALL * B
                eigvals, eigvecs = sclin.eigh(
                    A[np.ix_(indices, indices)], B[np.ix_(indices, indices)]
                )
                eigvecs_by_dir.append(np.real(eigvecs))
                eigvals_by_dir.append(np.real(eigvals))
            all_eigvecs_by_dir.append(eigvecs_by_dir)
            all_eigvals_by_dir.append(eigvals_by_dir)

        # Super class
        self._nonzeros_by_dir = nnz_by_dir
        self._indices_by_dir = indx_by_dir
        self._nbDoFsPerNode = nbDoFsPerNode

        # Self class
        self._space_dirs = space_dirs
        self._space_table_dirichlet = space_table_dirichlet
        self._eigenvec_by_dir_space = all_eigvecs_by_dir
        self._eigenval_by_dir_space = all_eigvals_by_dir
        self._mass_space_corrector = mass_corrector
        self._stiff_space_corrector = stif_corrector
        self._space_free_nodes = self._propagate_nodes_in_space()

    def update_space_eigenvalues(
        self, scalar_coefs: Sequence[float]
    ) -> List[np.ndarray]:
        """
        Update the spatial eigenvalues of the pencils (stiffness, mass) for each direction.

        Let say that the matrix is a linear combination of the mass and stiffness matrices (in the physical space):
            A = scalar_coefs[0] * M + scalar_coefs[1] * K.

        Then the preconditioner is built following the same linear combination but with the eigenvalues:
            P = scalar_coefs[0] * I + scalar_coefs[1] * Lambda.

        Here Lambda is the combination of the eigenvalues of the pencils (stiffness, mass) for each direction in space.

        Parameters
        ----------
        scalar_coefs : Sequence[float]
            Coefficients for the linear combination of mass and stiffness in the physical space.
            The first entry corresponds to the mass and the second entry corresponds to the stiffness.
        """
        space_eigenvalues = []

        if len(scalar_coefs) < 2:
            return space_eigenvalues

        for ii in range(self.nbDoFsPerNode):
            eigenvalues_mixed = combine_arrays(
                self.eigenval_by_dir_space[ii],
                self.stiff_space_corrector[ii],
            )
            current_eig = (
                scalar_coefs[0]
                * self.mass_space_corrector[ii]
                * np.ones_like(eigenvalues_mixed)
                + scalar_coefs[1] * eigenvalues_mixed
            )
            space_eigenvalues.append(current_eig)
        self._space_scalar_coefs = (float(scalar_coefs[0]), float(scalar_coefs[1]))
        return space_eigenvalues

    def add_scalar_space_correctors(
        self,
        mass_corrector: Optional[List[float]] = None,
        stiffness_corrector: Optional[List[np.ndarray]] = None,
    ):
        """
        Add scalar correctors that takes into account the geometry and material properties.

        It is important to remember that the (vanilla) preconditioner does not consider any information
        about geometry or material properties, then in order to improve the performance of the preconditioner,
        we can add some scalar correctors to the eigenvalues of the pencils (stiffness, mass) for each direction in space.

        Parameters
        ----------
        mass_corrector : List[float]
            List of scalar correctors for the mass matrix.
        stiffness_corrector : List[ndarray]
            List of scalar correctors for the stiffness matrix.
            Each entry should be an array of shape (nbdirs,) corresponding to the number of spatial directions.

        Notes
        -----
        The length of the list should be equal to the number of DoFs per node.
        """

        def is_verified(entry):
            return isinstance(entry, (list, tuple)) and len(entry) == self.nbDoFsPerNode

        if mass_corrector is not None and is_verified(mass_corrector):
            self._mass_space_corrector = mass_corrector
            self._space_eigenvalues = None
        if stiffness_corrector is not None and is_verified(stiffness_corrector):
            self._stiff_space_corrector = stiffness_corrector
            self._space_eigenvalues = None

    def apply_spatial_preconditioner(self, array_in: np.ndarray) -> np.ndarray:
        """
        Apply the spatial part of the fast-diagonalization preconditioner to the input array.

        Parameters
        ----------
        array_in : ndarray
            Input array to which the spatial preconditioner will be applied.

        Returns
        -------
        ndarray
            Output array after applying preconditioner.
        """
        start = time()
        tail_shape = array_in.shape[1:]
        array_in = np.reshape(array_in, (self.nbDoFsPerNode, -1, *tail_shape))
        array_out = np.zeros_like(array_in)

        for ii in range(self.nbDoFsPerNode):
            array_to_apply = array_in[ii][self.space_free_nodes[ii]]
            if np.linalg.norm(array_to_apply) == 0.0:
                continue

            array = MatrixFree.apply(
                self.eigenvec_by_dir_space[ii],
                array_to_apply,
                is_transpose=True,
            )
            eigenvalues = self.space_eigenvalues[ii].copy()
            if np.any(eigenvalues <= Constants.TINY):
                # Apply regularization
                logger.warning("Apply regularization")
                eigenvalues += Constants.TINY * np.max(eigenvalues)
            inv_eigenvalues = 1.0 / eigenvalues
            array = np.einsum("i,i...->i...", inv_eigenvalues, array, optimize=True)
            array_out[ii][self.space_free_nodes[ii]] = MatrixFree.apply(
                self.eigenvec_by_dir_space[ii], array, is_transpose=False
            )

        logger.debug(
            f"Single patch fast-diagonalization in {time() - start:.2e} seconds"
        )
        return np.reshape(array_out, (-1, *tail_shape))
