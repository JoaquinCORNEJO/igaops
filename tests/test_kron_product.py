import numpy as np
import pytest

from igaops.operators import MatrixFree
from igaops.common import KronProduct


def test_kron_nonzero_indices():
    vec1 = np.array([1, 0, 1, 0])
    vec2 = np.array([0, 1, 0, 1])
    vec3 = np.array([1, 1, 0])

    indices_ref = np.nonzero(np.kron(np.kron(vec1, vec2), vec3))[0]

    indices_list = [[0, 2], [1, 3], [0, 1]]
    nnz_list = [4, 4, 3]

    indices_app = np.array(KronProduct.kron_nonzero_indices(indices_list, nnz_list))

    np.testing.assert_array_equal(indices_app, indices_ref)


def test_kron_axis0_two_matrices():
    mat1 = np.array([[1, 2, 3], [2, 3, 4]])
    mat2 = np.array([[3, 4, 5], [4, 5, 6]])

    expected = np.array(
        [
            [3, 8, 15],
            [6, 12, 20],
            [4, 10, 18],
            [8, 15, 24],
        ]
    )

    result = KronProduct.kron_following_axis([mat1, mat2], axis=0)

    np.testing.assert_array_equal(result, expected)


def test_kron_axis0_two_matrices_matvec():
    rng = np.random.default_rng(0)
    v = rng.random(4)

    mat1 = np.array([[1, 2, 3], [2, 3, 4]])
    mat2 = np.array([[3, 4, 5], [4, 5, 6]])

    expected = v @ np.array(
        [
            [3, 8, 15],
            [6, 12, 20],
            [4, 10, 18],
            [8, 15, 24],
        ]
    )

    V = np.reshape(v, (2, 2), order="F")
    result = np.einsum("i...,j...,ij->...", mat1, mat2, V)

    np.testing.assert_array_equal(result, expected)

    result2 = KronProduct.mf_kron_following_axis([mat1, mat2], v, axis=0)

    np.testing.assert_array_equal(result2, expected)


def test_kron_axis1_two_matrices():
    mat1 = np.array([[1, 2], [2, 3], [3, 4]])
    mat2 = np.array([[3, 4], [4, 5], [5, 6]])

    expected = np.array([[3, 6, 4, 8], [8, 12, 10, 15], [15, 20, 18, 24]])

    result = KronProduct.kron_following_axis([mat1, mat2], axis=1)

    np.testing.assert_array_equal(result, expected)


def test_kron_axis1_two_matrices_matvec():
    rng = np.random.default_rng(0)
    v = rng.random(4)

    expected = np.array([[3, 6, 4, 8], [8, 12, 10, 15], [15, 20, 18, 24]]) @ v

    mat1 = np.array([[1, 2], [2, 3], [3, 4]])
    mat2 = np.array([[3, 4], [4, 5], [5, 6]])

    V = np.reshape(v, (2, 2), order="F")
    result = np.einsum("...i,...j,ij->...", mat1, mat2, V)

    np.testing.assert_array_equal(result, expected)

    result2 = KronProduct.mf_kron_following_axis([mat1, mat2], v, axis=1)

    np.testing.assert_array_equal(result2, expected)


def test_kron_axis0_three_matrices():
    mat1 = np.array([[1, 2, 3], [2, 3, 4]])
    mat2 = np.array([[3, 4, 5], [4, 5, 6]])
    mat3 = np.array([[1, 1, 1], [2, 2, 2]])

    expected = np.array(
        [
            [3, 8, 15],
            [6, 12, 20],
            [4, 10, 18],
            [8, 15, 24],
            [6, 16, 30],
            [12, 24, 40],
            [8, 20, 36],
            [16, 30, 48],
        ]
    )

    result = KronProduct.kron_following_axis([mat1, mat2, mat3], axis=0)

    np.testing.assert_array_equal(result, expected)


def test_kron_axis_single_matrix():
    mat1 = np.array([[1, 2, 3], [2, 3, 4]])

    result = KronProduct.kron_following_axis([mat1], axis=0)

    np.testing.assert_array_equal(result, mat1)

    result = KronProduct.kron_following_axis([mat1], axis=1)

    np.testing.assert_array_equal(result, mat1)


def test_kron_axis_single_matrix_matvec():
    rng = np.random.default_rng(0)
    v = rng.random(2)

    mat = np.array([[1, 2], [0, 3]])

    result = KronProduct.mf_kron_following_axis([mat], v, axis=0)
    np.testing.assert_array_equal(result, v @ mat)

    result = KronProduct.mf_kron_following_axis([mat], v, axis=1)
    np.testing.assert_array_equal(result, mat @ v)


def test_matrixfree_kron_matvec():
    A = np.array([[1, 2], [3, 6]])
    B = np.array([[5, 9, 6], [3, 2, 7]])
    C = np.array([[3, 2]])
    arr_ravel = np.random.default_rng(0).random(12)

    C_kron_B = np.kron(C, B)
    C_kron_B_kron_A = np.kron(C_kron_B, A)

    matvec_refe = C_kron_B_kron_A @ arr_ravel

    # Matrix free
    matvec_mf = MatrixFree.apply([A, B, C], arr_ravel)

    # Using Kronecker properties
    arr_reshaped = np.reshape(arr_ravel, (6, 2))
    arr_with_A = np.einsum("ij, kj->ki", A, arr_reshaped)
    matvec_kron = np.ravel(C_kron_B @ arr_with_A)

    np.testing.assert_allclose(matvec_mf, matvec_refe)
    np.testing.assert_allclose(matvec_kron, matvec_refe)
