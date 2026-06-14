import numpy as np
import pytest

from igaops.solvers import GCRODR, LinearSolver

NSIZE = 1000
NTEST = 4
M = 60  # krylov_size
K = 30  # recycled_size
TOL = 1e-6
MAXITERS = 500


def build_tridiagonal(n, a=2.05, b=-1.0):
    return (
        np.diag(np.full(n, a))
        + np.diag(np.full(n - 1, b), 1)
        + np.diag(np.full(n - 1, b), -1)
    )


@pytest.fixture(scope="module")
def apply_A():
    A = build_tridiagonal(NSIZE)
    return lambda x: A @ x


@pytest.fixture(scope="module")
def rhs_list(apply_A):
    x = np.zeros(NSIZE)
    x[0] = 1.0
    rhs = apply_A(x)
    return np.outer(np.linspace(0, 1, NTEST + 1)[1:], rhs)


@pytest.fixture(scope="module")
def linear_solver():
    return LinearSolver(tolerance=TOL, maxiters=MAXITERS)


@pytest.fixture(scope="module")
def gcrodr():
    return GCRODR(tolerance=TOL, maxiters=MAXITERS, krylov_size=M, recycled_size=K)


def test_linear_solver_converges(linear_solver, apply_A, rhs_list):
    output = linear_solver.solve(apply_A, rhs_list[0])
    assert output.residual.min() <= TOL


def test_gcrodr_converges(gcrodr, apply_A, rhs_list):
    output = gcrodr.solve(apply_A, rhs_list[0])
    assert output.residual.min() <= TOL


def test_gcrodr_matches_linear_solver(linear_solver, gcrodr, apply_A, rhs_list):
    x_ref = linear_solver.solve(apply_A, rhs_list[0]).solution
    x_gcrodr = gcrodr.solve(apply_A, rhs_list[0]).solution
    assert np.allclose(x_ref, x_gcrodr, atol=1e-4)


def test_gcrodr_true_residual_matches_reported(gcrodr, apply_A, rhs_list):
    output = gcrodr.solve(apply_A, rhs_list[0])
    true_res = np.linalg.norm(rhs_list[0] - apply_A(output.solution))
    assert true_res <= TOL * 10


@pytest.mark.parametrize("i", range(NTEST))
def test_sequential_solve_converges(gcrodr, apply_A, rhs_list, i):
    output = gcrodr.solve(apply_A, rhs_list[i])
    true_res = np.linalg.norm(rhs_list[i] - apply_A(output.solution))
    assert true_res <= TOL * 10
