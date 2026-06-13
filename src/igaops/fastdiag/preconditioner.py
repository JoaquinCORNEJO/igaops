from typing import List, Sequence, Optional, Union
import numpy as np

from igaops.quadrature import StandardGauss, WeightedQuadrature
from .utils.space_precond import SpaceFD
from .utils.time_arrow import SpTimeFD as FDarrowhead
from .utils.time_legacy import SpTimeFD as FDlegacy

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


class SingleFastDiagonalization:
    """
    This class implements the fast-diagonalization preconditioner for a single patch.
    """

    def __init__(self):
        self._space_fd: Optional[SpaceFD] = None
        self._sptime_fd: Optional[Union[FDarrowhead, FDlegacy]] = None

    @property
    def space_preconditioner(self):
        if self._space_fd is None:
            raise RuntimeError("Space preconditioner not defined.")
        return self._space_fd

    @property
    def sptm_preconditioner(self):
        if self._sptime_fd is None:
            raise RuntimeError("Space-time preconditioner not defined.")
        return self._sptime_fd

    def compute_space_eigendecomposition(
        self,
        space_quadrule_list: Sequence[QuadratureRule],
        space_table_dirichlet: np.ndarray,
    ):
        """
        Compute the eigendecomposition of the pencils (stiffness, mass)
        for each direction in space and for each DoFs per node.

        Parameters
        ----------
        space_quadrule_list : List[QuadratureRule]
            List of quadrature rules for each spatial direction.
        space_table_dirichlet : ndarray
            A boolean array indicating the Dirichlet boundary conditions for each DoFs per node and direction.
        """
        precond = SpaceFD()
        precond.compute_space_eigendecomposition(
            space_quadrule_list, space_table_dirichlet
        )
        self._space_fd = precond

    def compute_time_schurdecomposition(
        self, time_quadrule: QuadratureRule, use_arrowhead: bool = True
    ):
        """
        Compute the Schur decomposition of the pencil (advection, mass) for the time direction.

        Parameters
        ----------
        time_quadrule : QuadratureRule
            Quadrature rule for the time direction.
        use_arrowhead : bool
            If True, it uses ``arrow head`` preconditioner, otherwise the old method.

        Notes
        -----
        The old method is at least twice slower than arrow head.
        """
        if use_arrowhead:
            precond = FDarrowhead(self.space_preconditioner)
        else:
            precond = FDlegacy(self.space_preconditioner)
        precond.compute_time_schurdecomposition(time_quadrule)
        self._sptime_fd = precond

    def update_space_eigenvalues(self, scalar_coefs: Sequence[float]):
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
        self.space_preconditioner.update_space_eigenvalues(scalar_coefs)

    def add_scalar_space_time_correctors(
        self,
        mass_corrector: Optional[List[float]] = None,
        stiffness_corrector: Optional[List[np.ndarray]] = None,
        advection_corrector: Optional[List[float]] = None,
    ):
        """
        Add scalar correctors that takes into account the geometry and material properties.

        It is important to remember that the (vanilla) preconditioner does not consider any information
        about geometry or material properties, then in order to improve the performance of the preconditioner,
        we can add some scalar correctors to the eigenvalues of the pencils (stiffness, mass) for each direction in space
        and to the Schur decomposition of the pencil (advection, mass) for the time direction.

        Parameters
        ----------
        mass_corrector : List[float]
            List of scalar correctors for the mass matrix.
        stiffness_corrector : List[ndarray]
            List of scalar correctors for the stiffness matrix.
            Each entry should be an array of shape (nbdirs,) corresponding to the number of spatial directions.
        advection_corrector : List[float]
            List of scalar correctors for the advection matrix.

        Notes
        -----
        The length of the list should be equal to the number of DoFs per node.
        """
        if self._space_fd is not None:
            self._space_fd.add_scalar_space_correctors(
                mass_corrector=mass_corrector, stiffness_corrector=stiffness_corrector
            )
        if self._sptime_fd is not None:
            self._sptime_fd.add_scalar_time_correctors(
                advection_corrector=advection_corrector
            )

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
        return self.space_preconditioner.apply_spatial_preconditioner(array_in)

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
        return self.sptm_preconditioner.apply_spacetime_preconditioner(array_in)

    def apply_spacetime_preconditioner_trans(self, array_in: np.ndarray) -> np.ndarray:
        """
        Apply the transpose of full space-time fast-diagonalization preconditioner to the input array.

        Parameters
        ----------
        array_in : ndarray
            Input array to which the full space-time preconditioner will be applied

        Returns
        -------
        ndarray
            Output array after applying preconditioner.
        """
        return self.sptm_preconditioner.apply_spacetime_preconditioner_trans(array_in)

    def __repr__(self) -> str:
        message = f""""
            Fast diagonalization with
            {self.space_preconditioner.nbDoFsPerNode} DoFs per node
            Total number of nodes in space: {self.space_preconditioner.sp_nnz}
            Number of nodes free per DoFs (space): {[len(nodes) for nodes in self.space_preconditioner.space_free_nodes]}
        """

        if self._sptime_fd is not None:
            message += f"""
            Total number of nodes in time: {self.sptm_preconditioner.tm_nnz}
            Number of nodes free per DoFs (space-time): {[len(nodes) for nodes in self.sptm_preconditioner.sptm_free_nodes]}
            """
        return message
