from typing import List

from scipy.sparse.linalg import lsqr
from scipy import sparse as sp
from geomdl import helpers
import numpy as np

from igaops.common import Constants


class Operations:
    @staticmethod
    def increase_multiplicity_to_knotvector(
        repeat: int, degree: int, knotvector: np.ndarray
    ) -> np.ndarray:
        """Return a new knot vector with increased interior knot multiplicities.

        For every unique interior knot in ``knotvector``, its multiplicity in the
        returned knot vector will be increased by ``repeat``. The boundary knots
        (first and last ``degree + 1`` entries) are preserved.

        Parameters
        ----------
        repeat : int
            Number of additional repeats to add to each interior knot's
            multiplicity (non-negative).
        degree : int
            Degree of the spline. Used to identify the boundary knots.
        knotvector : array_like
            Input knot vector. Must contain at least ``2*(degree + 1)`` knots.

        Returns
        -------
        ndarray
            A sorted knot vector with increased multiplicities for the interior
            knots.
        """
        kv_unique = np.unique(knotvector)
        is_first_or_last = np.ones_like(kv_unique, dtype=bool)
        is_first_or_last[1:-1] = False

        kv_out = []
        for ifl, knot in zip(is_first_or_last, kv_unique):
            r = 0 if ifl else repeat
            m = helpers.find_multiplicity(knot, knotvector) + r
            if m > degree + 1:
                raise Warning("Introduces numerical instability.")
            kv_out.extend([knot] * m)

        return np.sort(kv_out)

    @staticmethod
    def eval_ders_basis_sparse(
        degree: int,
        knotvector: np.ndarray,
        knots: np.ndarray,
        nders: int = 1,
    ) -> List[sp.csr_array]:
        """
        Evaluate B-spline basis functions and their derivatives at the given knots.
        The basis values and derivatives are returned in COO sparse-matrix format.
        If there are ``n`` basis functions and ``m`` evaluation knots, each
        returned matrix has shape ``(n, m)``.

        Parameters
        ----------
        degree : int
            Degree of the spline.
        knotvector : array_like
            Knot vector defining the spline.
        knots : array_like
            Evaluation knots.
        nders : int, optional
            Number of derivatives to compute. Default is ``1``.

        Returns
        -------
        List[sparray]
            List of sparse matrices in CSR format.
        """

        nbctrlpts = len(knotvector) - degree - 1
        values, indices_i, indices_j = [], [], []

        for j, knot in enumerate(knots):
            knot_span = helpers.find_span_linear(degree, knotvector, nbctrlpts, knot)
            basis_ders = helpers.basis_function_ders(
                degree, knotvector, knot_span, knot, nders
            )

            for i, output in enumerate(zip(*basis_ders)):
                values.append(output)
                indices_i.append(knot_span - degree + i)
                indices_j.append(j)
        values = np.asarray(values)

        # Loop through the basis functions and derivatives
        basis_list = []
        for i in range(values.shape[1]):
            b = sp.coo_array(
                (values[:, i], (indices_j, indices_i)), shape=(len(knots), nbctrlpts)
            )
            b.eliminate_zeros()
            basis_list.append(sp.csr_array(b))
        return basis_list

    @staticmethod
    def solve_optimization_problem(
        Z: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """
        Solves the optimization problem
            Minimize ||diag(Z_i)^{-1} @  w||^2.

        Subject to
            A @ w = B_i, for i = 1, ..., n.

        Parameters
        ----------
        Z : array_like
            Coefficient matrix of size (n, m).
        A : array_like
            Constraint matrix of size (p, m).
        B : array_like
            Right-hand side matrix of size (n, p).

        Returns
        -------
        ndarray
            Solution vector w of size (n, m,).
        """
        assert B.shape[0] == Z.shape[0], "No. columns in b must match the no. rows in z"
        assert B.shape[1] == A.shape[0], "B and A must have compatible dimensions"
        assert A.shape[1] == Z.shape[1], "A and Z must have compatible dimensions"

        tol = Constants.SAFEGUARD
        sol = np.zeros_like(Z)
        for ii, (zz, bb) in enumerate(zip(Z, B)):
            sol[ii] = zz * lsqr(A * zz, bb, atol=tol, btol=tol)[0]

        return sol
