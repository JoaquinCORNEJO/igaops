from typing import Tuple
import numpy as np

from igaops.common import Constants


class Operations:
    @staticmethod
    def make_open_knotvector(
        degree: int, nbel: int, multiplicity: int = 1
    ) -> np.ndarray:
        """
        Creates an open and uniform knotvector.

        Parameters
        ----------
        degree : int
            Polynomial degree of spline.
        nbel : int
            Number of elements of spline
        multiplicity : int, optional
            Multiplicity of inner knots in knot-vector. Defaults to 1.
        """
        return np.concatenate(
            (
                np.zeros(degree + 1),
                np.repeat(np.linspace(0.0, 1.0, nbel + 1)[1:-1], multiplicity),
                np.ones(degree + 1),
            )
        )

    @staticmethod
    def make_closed_knotvector(degree: int, nbel: int):
        """
        Creates a closed and uniform knot-vector.

        Parameters
        ----------
        degree : int
            Polynomial degree of spline.
        nbel : int
            Number of elements of spline
        """
        ukv = np.linspace(0.0, 1.0, nbel + 1)
        m = degree // nbel + 1
        kv_right = ukv[1:] + 1.0
        kv_left = ukv[-2::-1] - 1.0
        # Case where nbel <= degree
        for i in range(1, m):
            kv_right = np.concatenate((kv_right, kv_right[:nbel] + i))
            kv_left = np.concatenate((kv_left, kv_left[:nbel] - i))
        return np.concatenate((kv_left[:degree][::-1], ukv, kv_right[:degree]))

    @staticmethod
    def evaluate_greville(degree: int, knotvector: np.ndarray):
        """
        Computes the Greville points from knot-vector.

        Parameters
        ----------
        degree : int
            Polynomial degree of spline.
        knotvector : array_like
            Knot-vector of spline.
        """
        return np.array(
            [
                sum(knotvector[i + j + 1] for j in range(degree)) / degree
                for i in range(0, len(knotvector) - degree - 1)
            ]
        )

    @staticmethod
    def eval_inverse_and_determinant(
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the determinant and the inverse of a square matrix.

        Parameters
        ----------
        matrix : ndarray
            matrix of size m x m x ..., with m <= 3.

        Returns
        -------
        tuple[ndarray,ndarray]
            A tuple with the determinant and inverse of the matrix.

        Raises
        ------
        NotImplementedError
            If m > 3 with m being the size of the matrix.
        ZeroDivisionError
            If any value of the determinant is close to zero.
        """
        matrix = np.asarray(matrix).copy()
        m1, _ = matrix.shape[:2]
        tail_shape = matrix.shape[2:]
        inv = np.zeros(shape=(m1, m1) + tail_shape)
        if m1 == 1:
            det = np.copy(matrix[0][0])
            inv = np.ones_like(inv)
        elif m1 == 2:
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            inv[0][0] = matrix[1][1]
            inv[1][1] = matrix[0][0]
            inv[0][1] = -matrix[0][1]
            inv[1][0] = -matrix[1][0]
        elif m1 == 3:
            det = (
                matrix[0][1] * matrix[1][2] * matrix[2][0]
                - matrix[0][2] * matrix[1][1] * matrix[2][0]
                + matrix[0][2] * matrix[1][0] * matrix[2][1]
                - matrix[0][0] * matrix[1][2] * matrix[2][1]
                + matrix[0][0] * matrix[1][1] * matrix[2][2]
                - matrix[0][1] * matrix[1][0] * matrix[2][2]
            )
            inv[0][0] = matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]
            inv[0][1] = matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]
            inv[0][2] = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]
            inv[1][0] = matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]
            inv[1][1] = matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]
            inv[1][2] = matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]
            inv[2][0] = matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]
            inv[2][1] = matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]
            inv[2][2] = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            raise NotImplementedError("Only 1x1, 2x2 and 3x3 matrices are supported")

        if np.any(np.abs(det) < Constants.SAFEGUARD):
            raise ZeroDivisionError("There are near to zero determinants")

        inv = inv / det
        return det, inv

    @staticmethod
    def inverse_rectangular_matrix(jac: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Moore-Penrose pseudoinverse of a rectangular matrix A (here calle jac).

        Parameters
        ----------
        jac : np.ndarray
            Input matrix of shape (m, n, ...)

        Returns
        -------
        tuple[ndarray,ndarray]
            - Determinant-like of jac, shape (...)
            - Pseudo-inverse of jac, shape (n, m, ...)
        """
        # Metric tensor G = J^T J over param space
        G = np.einsum("li...,lj...->ij...", jac, jac, optimize=True)
        det_G, inv_G = Operations.eval_inverse_and_determinant(G)
        det_jac = np.sqrt(det_G)
        # J^{-1}_phys = inv(J^T J) J^T
        inv_jac = np.einsum("il...,jl...->ij...", inv_G, jac, optimize=True)
        return det_jac, inv_jac
