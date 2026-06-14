import numpy as np
import pytest

from igaops.quadrature import StandardGauss
from igaops.operators import BsplineOperations, MatrixFree
from igaops.geometry import GeometryOps
from igaops.fastdiag import SingleFD
from igaops.solvers import LinearSolver


@pytest.fixture(params=[(1, 3, 4), (2, 3, 4), (3, 3, 4)])
def problem(request):
    """
    Parameters are (NDIM, DEGREE, NBEL).
    """
    ndim, degree, nbel = request.param

    knotvector = GeometryOps.make_open_knotvector(degree, nbel)

    quadrature = StandardGauss(
        degree=degree,
        knotvector=knotvector,
        quadtype="legendre",
    )

    quadrule_list = [quadrature] * ndim

    nnz = np.prod([q.nbquadpts for q in quadrule_list])

    return {
        "ndim": ndim,
        "degree": degree,
        "nbel": nbel,
        "quadrule_list": quadrule_list,
        "nnz": nnz,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_mass_matrix_matrix_free(problem, rng):
    quadrule_list = problem["quadrule_list"]
    nnz = problem["nnz"]

    coefficients = np.ones(nnz)

    mass = BsplineOperations.assemble_scalar_u_v(
        quadrule_list=quadrule_list,
        coefficients=coefficients,
    )

    v = rng.random(mass.shape[0])

    mv_dense = mass @ v

    mv_mf = BsplineOperations.compute_mf_scalar_u_v(
        quadrule_list=quadrule_list,
        coefficients=coefficients,
        array_in=v,
    )

    np.testing.assert_allclose(mv_dense, mv_mf)


def test_matrixfree_dense_sparse(problem, rng):
    quadrule_list = problem["quadrule_list"]

    matrix_list = [q.basis[0].toarray() for q in quadrule_list]

    n = np.prod([A.shape[1] for A in matrix_list])
    v = rng.random(n)

    y_dense = MatrixFree.apply(
        matrix_list=matrix_list,
        array_in=v,
        is_transpose=False,
    )

    y_sparse = MatrixFree.apply(
        matrix_list=[q.basis[0] for q in quadrule_list],
        array_in=v,
        is_transpose=False,
    )

    np.testing.assert_allclose(y_dense, y_sparse)


def test_matrixfree_matches_kronecker(problem, rng):
    quadrule_list = problem["quadrule_list"]

    matrix_list = [q.basis[0].toarray() for q in quadrule_list]

    n = np.prod([A.shape[1] for A in matrix_list])
    v = rng.random(n)

    y_mf = MatrixFree.apply(
        matrix_list=matrix_list,
        array_in=v,
        is_transpose=False,
    )

    kron_matrix = matrix_list[0]
    for A in matrix_list[1:]:
        kron_matrix = np.kron(A, kron_matrix)

    y_kron = kron_matrix @ v

    np.testing.assert_allclose(y_mf, y_kron)


def build_fd(quadrule_list, ndim):
    table_dirichlet = np.ones((1, ndim, 2), dtype=bool)
    table_dirichlet[0, 0, 0] = False

    fd = SingleFD()

    fd.compute_space_eigendecomposition(
        space_quadrule_list=quadrule_list,
        space_table_dirichlet=table_dirichlet,
    )

    fd.update_space_eigenvalues([0, 1])

    return fd


def test_fast_diagonalization(problem, rng):
    ndim = problem["ndim"]
    quadrule_list = problem["quadrule_list"]

    fd = build_fd(quadrule_list, ndim)

    free_nodes = fd.space_preconditioner.space_free_nodes[0]
    U_list = fd.space_preconditioner.eigenvec_by_dir_space[0]
    eigenvalues = fd.space_preconditioner.space_eigenvalues[0]

    size = np.max(free_nodes) + 1
    v = rng.random(size)

    UU = U_list[0]
    for U in U_list[1:]:
        UU = np.kron(U, UU)

    tmp = UU.T @ v[free_nodes]
    tmp /= eigenvalues
    tmp = UU @ tmp

    pv_dense = np.zeros_like(v)
    pv_dense[free_nodes] = tmp

    pv_mf = fd.apply_spatial_preconditioner(v)

    np.testing.assert_allclose(pv_dense, pv_mf)


def test_preconditioner_matches_direct_solver(problem, rng):
    ndim = problem["ndim"]
    quadrule_list = problem["quadrule_list"]
    nnz = problem["nnz"]

    fd = build_fd(quadrule_list, ndim)

    free_nodes = fd.space_preconditioner.space_free_nodes[0]

    coefficients = np.zeros((ndim, ndim, nnz))
    for i in range(ndim):
        coefficients[i, i, :] = 1.0

    stiffness = BsplineOperations.assemble_scalar_gradu_gradv(
        quadrule_list=quadrule_list,
        coefficients=coefficients,
    )

    size = stiffness.shape[0]
    v = rng.random(size)

    sol_mf = fd.apply_spatial_preconditioner(v)

    A = stiffness[np.ix_(free_nodes, free_nodes)]
    b = v[free_nodes]

    sol_dense = np.zeros_like(v)
    sol_dense[free_nodes] = LinearSolver.direct(A, b).solution

    np.testing.assert_allclose(sol_mf, sol_dense, atol=1e-5)
