from typing import Any, Dict, List
import numpy as np
import pytest
import scipy.linalg as sclin
import scipy.sparse as sp

from igaops.common import combine_arrays
from igaops.geometry import GeometryOps
from igaops.fastdiag import SingleFD
from igaops.operators import BsplineOperations
from igaops.quadrature import StandardGauss
from igaops.solvers import LinearSolver

NDIM = 3
DEGREE = 3
NBEL = 4
VALS = [3.0, 2.0, 5.0]


@pytest.fixture(scope="module")
def setup() -> Dict[str, Any]:
    """Build all data shared by the tests."""

    knotvector = GeometryOps.make_open_knotvector(DEGREE, NBEL)

    quadrature = StandardGauss(
        degree=DEGREE,
        knotvector=knotvector,
        quadtype="legendre",
    )

    weights: List[sp.csr_array] = quadrature.weights
    basis: List[sp.csr_array] = quadrature.basis

    idx_1 = np.arange(1, quadrature.nbctrlpts)
    MM_1 = (weights[0].dot(basis[0])).toarray()[np.ix_(idx_1, idx_1)]
    KK_1 = (weights[-1].dot(basis[-1])).toarray()[np.ix_(idx_1, idx_1)]
    eigval_1, eigvec_1 = sclin.eigh(KK_1, MM_1)
    ones_1 = np.ones_like(eigval_1)

    idx_2 = np.arange(0, quadrature.nbctrlpts)
    MM_2 = (weights[0].dot(basis[0])).toarray()[np.ix_(idx_2, idx_2)]
    KK_2 = (weights[-1].dot(basis[-1])).toarray()[np.ix_(idx_2, idx_2)]
    eigval_2, eigvec_2 = sclin.eigh(KK_2, MM_2)
    ones_2 = np.ones_like(eigval_2)

    idx_3 = np.arange(0, quadrature.nbctrlpts - 1)
    MM_3 = (weights[0].dot(basis[0])).toarray()[np.ix_(idx_3, idx_3)]
    KK_3 = (weights[-1].dot(basis[-1])).toarray()[np.ix_(idx_3, idx_3)]
    eigval_3, eigvec_3 = sclin.eigh(KK_3, MM_3)
    ones_3 = np.ones_like(eigval_3)

    eigenvalues_ref = (
        VALS[0] * np.kron(np.kron(ones_3, ones_2), eigval_1)
        + VALS[1] * np.kron(np.kron(ones_3, eigval_2), ones_1)
        + VALS[2] * np.kron(np.kron(eigval_3, ones_2), ones_1)
    )

    eigenvector_ref = np.kron(
        np.kron(eigvec_3, eigvec_2),
        eigvec_1,
    )

    quadrule_list = [quadrature] * NDIM

    table_dirichlet = np.zeros((1, NDIM, 2), dtype=bool)
    table_dirichlet[0, 0, 0] = True
    table_dirichlet[0, 2, 1] = True

    fd = SingleFD()
    fd.compute_space_eigendecomposition(
        space_quadrule_list=quadrule_list,
        space_table_dirichlet=table_dirichlet,
    )

    fd.add_scalar_space_time_correctors(stiffness_corrector=[np.array(VALS)])
    fd.update_space_eigenvalues(scalar_coefs=(0, 1))

    free_nodes = fd.space_preconditioner.space_free_nodes[0]

    U_list = fd.space_preconditioner.eigenvec_by_dir_space[0]

    eigenvalues = fd.space_preconditioner.space_eigenvalues[0]

    other_eigenvalues = combine_arrays(
        fd.space_preconditioner.eigenval_by_dir_space[0],
        np.array(VALS),
    )

    UU = U_list[0]
    for current in U_list[1:]:
        UU = np.kron(current, UU)

    nnz_ctrlpts = np.prod([q.nbctrlpts for q in quadrule_list])

    nnz_quadpts = np.prod([q.nbquadpts for q in quadrule_list])

    coefficients = np.zeros((NDIM, NDIM, nnz_quadpts))
    for i, value in enumerate(VALS):
        coefficients[i, i, :] = value

    stiffness = BsplineOperations.assemble_scalar_gradu_gradv(
        quadrule_list=quadrule_list,
        coefficients=coefficients,
    )

    return {
        "fd": fd,
        "quadrule_list": quadrule_list,
        "free_nodes": free_nodes,
        "eigenvalues": eigenvalues,
        "other_eigenvalues": other_eigenvalues,
        "eigenvalues_ref": eigenvalues_ref,
        "UU": UU,
        "eigenvector_ref": eigenvector_ref,
        "coefficients": coefficients,
        "STIFFNESS": stiffness,
        "nnz_ctrlpts": nnz_ctrlpts,
    }


@pytest.fixture(scope="module")
def preconditioner(setup):
    eigenvector_ref = setup["eigenvector_ref"]
    eigenvalues_ref = setup["eigenvalues_ref"]

    def apply(x: np.ndarray) -> np.ndarray:
        tmp = eigenvector_ref.T @ x
        tmp /= eigenvalues_ref
        return eigenvector_ref @ tmp

    return apply


@pytest.fixture(scope="module")
def stiffness_matrix(setup):
    free_nodes = setup["free_nodes"]

    return setup["STIFFNESS"][np.ix_(free_nodes, free_nodes)]


@pytest.fixture(scope="module")
def matrix_free_matvec(setup):
    quadrule_list = setup["quadrule_list"]
    free_nodes = setup["free_nodes"]
    coefficients = setup["coefficients"]

    def apply(x: np.ndarray) -> np.ndarray:
        nnz = np.prod([q.nbctrlpts for q in quadrule_list])

        arr_in = np.zeros(nnz)
        arr_in[free_nodes] = x

        arr_out = BsplineOperations.compute_mf_scalar_gradu_gradv(
            quadrule_list=quadrule_list,
            coefficients=coefficients,
            array_in=arr_in,
        )

        return arr_out[free_nodes]

    return apply


@pytest.fixture(scope="module")
def solver_results(
    setup,
    stiffness_matrix,
    matrix_free_matvec,
    preconditioner,
):
    free_nodes = setup["free_nodes"]

    rng = np.random.default_rng(0)
    b = rng.random(len(free_nodes))

    linear_solver = LinearSolver(
        tolerance=1e-9,
        maxiters=100,
        linear_type="gmres",
    )

    output = linear_solver.solve(
        Afun=matrix_free_matvec,
        b=b,
        Pfun=preconditioner,
    )

    direct_solution = LinearSolver.direct(
        stiffness_matrix,
        b,
    ).solution

    niter = linear_solver.convergence_manager.parameters["iteration"].current

    return {
        "b": b,
        "iterative_solution": output.solution,
        "direct_solution": direct_solution,
        "niter": niter,
    }


def test_eigenvalues_match_reference(
    setup: Dict[str, Any],
):
    np.testing.assert_allclose(
        setup["eigenvalues"], setup["eigenvalues_ref"], rtol=1e-4
    )


def test_combined_eigenvalues_match_reference(
    setup: Dict[str, Any],
):
    np.testing.assert_allclose(
        setup["other_eigenvalues"], setup["eigenvalues_ref"], rtol=1e-4
    )


def test_eigenvectors_match_reference(
    setup: Dict[str, Any],
):
    np.testing.assert_allclose(setup["UU"], setup["eigenvector_ref"], atol=1e-10)


def test_matrix_free_matvec_matches_dense(
    stiffness_matrix,
    matrix_free_matvec,
):
    rng = np.random.default_rng(0)
    b = rng.random(stiffness_matrix.shape[0])

    np.testing.assert_allclose(
        stiffness_matrix @ b,
        matrix_free_matvec(b),
    )


def test_iterative_solution_matches_direct(
    solver_results,
):
    np.testing.assert_allclose(
        solver_results["iterative_solution"],
        solver_results["direct_solution"],
    )


def test_gmres_converges_in_few_iterations(
    solver_results,
):
    assert solver_results["niter"] < 5


def test_spatial_preconditioner_matches_dense_reference(
    setup,
    preconditioner,
):
    fd = setup["fd"]
    free_nodes = setup["free_nodes"]

    rng = np.random.default_rng(0)
    v = rng.random(setup["nnz_ctrlpts"])

    Pv_dense = np.zeros_like(v)
    Pv_dense[free_nodes] = preconditioner(v[free_nodes])

    Pv_mf = fd.apply_spatial_preconditioner(v)

    # NOTE: the relative error is "high" because FD adds regularizations
    # to avoid division by zero
    np.testing.assert_allclose(Pv_dense, Pv_mf, rtol=1e-2)
