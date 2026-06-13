from typing import Sequence, List, Optional
from itertools import product
from functools import reduce
from operator import mul
import string

import numpy as np


def combine_arrays(
    arr: Sequence[np.ndarray],
    coefs: np.ndarray,
    brr: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """
    Computes a linear combination using kron products.

    It acutally computes:
        sum_{i=0 to n} c_i * (b_n x ... x a_i x ... x b_1)
    where c_i are scalar coeficients and a_i, b_i are arrays.

    Parameters
    ----------
    arr : Sequence[ndarray]
        List of arrays from where we define a_i arrays.
    coefs : array_like
        List of coefficients from where we define c_i scalars.
    brr : Sequence[ndarray], optional
        List of arrays from where we define b_i arrays. If not defined, ones are used.

    Returns
    -------
    ndarray
        Linear combination.
    """

    def kron(arrays: Sequence[np.ndarray]) -> np.ndarray:
        a = arrays[0]
        for curr in arrays[1:]:
            a = np.kron(curr, a)
        return a

    ndim = len(arr)

    if ndim == 0:
        return np.array([])

    if brr is None:
        brr = [np.ones_like(a) for a in arr]

    mixed = [[brr[_] for _ in range(ndim)] for _ in range(ndim)]
    for i in range(ndim):
        mixed[i][i] = arr[i] * coefs[i]

    v_out = kron(mixed[0])
    for i in range(1, ndim):
        v_out += kron(mixed[i])
    return v_out


class KronProduct:
    @staticmethod
    def kron_nonzero_indices(
        indices_list: Sequence[List[int]], nnz_list: Sequence[int]
    ) -> List[int]:
        """
        Finds the nonzero indices of a kronecker product of sparse arrays.

        For example, let say A_1, A_2, ..., A_n are sparse arrays, then
        the result A = A_n x ... x A_2 x A_1 is also sparse. It computes then
        the nonzero values of A knowing the nonzero values of A_1, ..., A_n.

        Parameters
        ----------
        indices_list : Sequence[List[int]]
            List that constains the nonzero indices of the different arrays.
        nnz_list : Sequence[int]
            List that contains the arrays' size.

        Returns
        -------
        List[int]
            A list of nonzero indices of the resulting array
        """
        strides = [reduce(mul, nnz_list[i + 1 :], 1) for i in range(len(nnz_list))]
        flat_indices = []
        for multi_idx in product(*indices_list):
            flat = sum(i * s for i, s in zip(multi_idx, strides))
            flat_indices.append(flat)
        return flat_indices

    @staticmethod
    def kron_following_axis(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        """
        Perfom Kron product between matrices ofollowing axis 0 or 1.

        Parameters
        ----------
        arrays : Sequence[ndarray]
            List of arrays to perform method.
        axis : int
            Axis along the kron product is performed while the other is fixed.

        Returns
        -------
        ndarray

        Example
        -------
        Let say
        mat1 = np.array([[1, 2, 3], [2, 3, 4]])
        mat2 = np.array([[3, 4, 5], [4, 5, 6]])

        The result following axis 0 should be:
        np.array(
            [
                [3, 8, 15],
                [6, 12, 20],
                [4, 10, 18],
                [8, 15, 24],
            ]
        )
        """
        if not arrays:
            raise ValueError("Empty list")

        if axis not in [0, 1]:
            raise ValueError("Only perform following axis 0 or 1")

        if len(arrays) == 1:
            return arrays[0]

        result = arrays[0]
        n, m = arrays[0].shape

        if axis == 0:
            for curr in arrays[1:]:
                result = np.reshape(
                    curr[None, :, :] * result[:, None, :], (-1, m), order="F"
                )
        elif axis == 1:
            for curr in arrays[1:]:
                result = np.reshape(
                    curr[:, None, :] * result[:, :, None], (n, -1), order="F"
                )
        else:
            raise NotImplementedError()

        return result

    @staticmethod
    def mf_kron_following_axis(
        arrays: Sequence[np.ndarray], array_in: np.ndarray, axis: int = 0
    ) -> np.ndarray:
        """
        Perfom matrix-free Kron product between matrices and array.

        Parameters
        ----------
        arrays : Sequence[ndarray]
            List of arrays to perform method.
        array_in : ndarray
            Input array to perform matrix-free.
        axis : int
            Axis along the kron product is performed while the other is fixed.

        Returns
        -------
        ndarray
        """
        if not arrays:
            raise ValueError("Empty list")

        if axis not in [0, 1]:
            raise ValueError("Only perform following axis 0 or 1")

        labels = string.ascii_lowercase[: len(arrays)]
        rhs = "".join(labels)

        if axis == 0:
            lhs = ",".join(f"{char}..." for char in labels)
        elif axis == 1:
            lhs = ",".join(f"...{char}" for char in labels)
        else:
            raise NotImplementedError()

        new_shape = [arr.shape[axis] for arr in arrays]
        arr_reshaped = np.reshape(array_in, new_shape, order="F")
        einsumtext = f"{lhs},{rhs}->..."
        return np.einsum(einsumtext, *arrays, arr_reshaped, optimize=True)
