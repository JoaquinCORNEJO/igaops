from typing import Union, Sequence

from scipy import sparse as sp
import numpy as np


class MatrixFree:

    @staticmethod
    def reshape_array_to_tensor(
        matrix_list: Sequence[Union[sp.csr_array, np.ndarray]],
        array_in: np.ndarray,
        is_transpose: bool = False,
    ) -> np.ndarray:
        """
        Reshape the input array into a tensor with dimensions corresponding to the number of matrices in the matrix_list.

        Parameters
        ----------
        matrix_list : Sequence[array_like]
            A list of sparse or dense matrices [M_1, M_2, ..., M_n].
        array_in : ndarray
            The input array to be reshaped into a tensor.
        is_transpose : bool
            If true, the dimensions of the tensor are determined based on the transpose of the matrices.

        Returns
        -------
        ndarray
            The reshaped tensor with dimensions corresponding to the number of matrices in the matrix_list.
        """
        # We reverse the order to match Mn x ... x M1
        original_shape = [
            matrix.shape[0 if is_transpose else 1] for matrix in matrix_list
        ]
        original_shape.extend(array_in.shape[1:] or (1,))
        return np.reshape(array_in, original_shape)

    @staticmethod
    def tensor_matrix_product(
        tensor: np.ndarray,
        matrix: Union[sp.csr_array, np.ndarray],
        mode: int,
        is_transpose: bool = False,
    ) -> np.ndarray:
        """
        Perform the n-mode product of a tensor with a matrix along a specified mode.

        Parameters
        ----------
        tensor : ndarray
            The input tensor to be multiplied with the matrix.
        matrix : array_like
            The matrix to be multiplied with the tensor.
        mode : int
            The mode along which to perform the multiplication.
        is_transpose : bool
            If true, the multiplication is performed with the transpose of the matrix.

        Returns
        -------
        ndarray
            The result of the n-mode product of the tensor with the matrix.
        """

        mat = matrix.T if is_transpose else matrix

        old_shape = tensor.shape
        new_shape = [mat.shape[0]] + [
            old_shape[i] for i in range(len(old_shape)) if i != mode
        ]

        tensor_perm = np.reshape(np.moveaxis(tensor, mode, 0), (old_shape[mode], -1))
        new_tensor = np.reshape(mat @ tensor_perm, new_shape)
        return np.moveaxis(new_tensor, 0, mode)

    @staticmethod
    def apply(
        matrix_list: Sequence[Union[sp.csr_array, np.ndarray]],
        array_in: np.ndarray,
        is_transpose: bool = False,
    ) -> np.ndarray:
        """
        Computes the matrix-free product M @ v, with M being the result of
        (M_n x ... x M_2 x M_1), where 'x' represents Kronecker product and
        M_i are 2-dimensional matrices.
        Note that (M_n x ... x M_2 x M_1) . v = V x M_n x_n M_(n-1) x_(n-1) ... x_1 M_1
        where v is a ravel of V and x_i is the i-mode tensor product

        Parameters
        ----------
        matrix_list : Sequence[array_like]
            A list of sparse of dense matrices [M_1, M_2, ..., M_n].
        array_in : ndarray
            The vector to be multiplied with the resulting matrix.
        is_transpose : bool
            If true it performs transpose(M) @ v, if false M @ v. By default is set to false.

        Returns
        -------
        ndarray
            The result of the matrix-vector product.
        """
        matrices = matrix_list[::-1]
        tensor = MatrixFree.reshape_array_to_tensor(
            matrices, array_in, is_transpose=is_transpose
        )
        for i, matrix in enumerate(matrices):
            # NOTE: the mode is i and not (nmodes - i - 1) due to C-ordering
            # resulting on a match with Kronecker product. Otherwise, we should
            # reshape the tensor using F-ordering which is not native in python
            # leading to unnecesary copies to access memory.
            tensor = MatrixFree.tensor_matrix_product(
                tensor, matrix, mode=i, is_transpose=is_transpose
            )

        if array_in.ndim == 1:
            return tensor.ravel()
        return tensor.reshape((-1,) + array_in.shape[1:])
