from typing import Sequence, Tuple

from scipy import sparse as sp
import numpy as np

from .matrix_free import MatrixFree


class Operations:
    @staticmethod
    def process_layers(
        matrix_list: Sequence[Sequence[sp.csr_array]],
        tensor: np.ndarray,
        is_transpose: bool,
    ):
        """
        Process the layers of the sum-factorization algorithm using a stack-based approach.

        Parameters
        ----------
        matrix_list : List[List[csr_array]]
            A list of lists of sparse matrices, where each inner list corresponds to a layer of the sum-factorization algorithm.
        tensor : ndarray
            The input tensor to be processed through the layers.
        is_transpose : bool
            A flag indicating whether to use the transpose of the matrices in the matrix_list.

        Returns
        -------
        Tuple[List[ndarray], List[ndarray]]
            A tuple containing two lists: the first list contains the non-zero data values from the final layer,
            and the second list contains the corresponding row indices for those non-zero values.
        """
        stack: Sequence[Tuple[int, np.ndarray]] = []
        data_list: Sequence[np.ndarray] = []
        rows_list: Sequence[np.ndarray] = []
        stack.append((0, tensor))
        while stack:
            layer, current_tensor = stack.pop()

            # Finished all layers
            if layer == len(matrix_list):
                # Only recover non-zero values
                dat = current_tensor.ravel()
                nnz = np.nonzero(dat)[0]
                data_list.append(dat[nnz])
                rows_list.append(nnz)
                continue

            # Expand next layer
            for mat in matrix_list[layer]:
                next_tensor = MatrixFree.tensor_matrix_product(
                    current_tensor, mat, layer, is_transpose=is_transpose
                )
                stack.append((layer + 1, next_tensor))

        return data_list[::-1], rows_list[::-1]

    @staticmethod
    def sumfact_assembling(
        weights: Sequence[sp.csr_array],
        basis: Sequence[sp.csr_array],
        coefficients: np.ndarray,
    ) -> sp.csr_array:
        """
        Assembles the global matrix using the sum-factorization technique.

        Parameters
        ----------
        weights : List[csr_array]
            A list of sparse matrices representing the weights for each dimension.
        basis : List[csr_array]
            A list of sparse matrices representing the basis functions for each dimension.
        coefficients : ndarray
            Contains geometry and material properties. It is a 0-rank tensor field.

        Returns
        -------
        csr_array
            The assembled global matrix in sparse format.
        """
        # NOTE: we assume that weights W_i are nr_i x nq_i while basis B_i are nq_i x nc_i
        nr_list = [w.shape[0] for w in weights]
        nc_list = [b.shape[1] for b in basis]
        total_nr = np.prod(nr_list)
        total_nc = np.prod(nc_list)

        # Precompute W_i * B_i[:, j] for j = 0, ... nc_i and for all i = 0, ... d
        basis_broadcasted = []
        for w, b, nc in zip(weights, basis, nc_list):
            b = b.tocsc()  # fast column access
            mats = [w.multiply(b[:, j]) for j in range(nc)]
            for m in mats:
                m.eliminate_zeros()
            basis_broadcasted.append(mats)

        # Sum-factorization
        tensor = MatrixFree.reshape_array_to_tensor(
            basis[::-1], coefficients, is_transpose=True
        )
        data_chunks, rows_chunks = Operations.process_layers(
            basis_broadcasted[::-1], tensor, is_transpose=False
        )
        cols_chunks = []
        for idx, nz in enumerate(rows_chunks):
            cols_chunks.append(np.full(nz.size, idx, dtype=np.int64))

        # Save results
        data = np.concatenate(data_chunks)
        rows = np.concatenate(rows_chunks)
        cols = np.concatenate(cols_chunks)

        mat = sp.csr_array((data, (rows, cols)), shape=(total_nr, total_nc))
        mat.eliminate_zeros()
        return mat
