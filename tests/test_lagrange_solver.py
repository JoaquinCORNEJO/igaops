import numpy as np
import pytest

from igaops.solvers import NonLinearSolver, LagrangeSolver


@pytest.fixture
def Cmat():
    return np.array([[0.0, 1.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


@pytest.fixture
def fvec():
    return np.array([0.0, 0.0, 0.0, 10])


@pytest.fixture
def gvec():
    return np.array([0.0, 0.0])


def matvec(u):
    k = 100.0
    A = np.array([[k, -k, 0, 0], [-k, k, 0, 0], [0, 0, k, -k], [0, 0, -k, k]])
    Au = np.asarray(A @ u)
    return Au


def test_direct_solver(Cmat, fvec, gvec):
    Amat = matvec(np.eye(len(fvec)))
    Zmat = np.zeros((len(gvec), len(gvec)))
    matrix = np.block([[Amat, Cmat.T], [Cmat, Zmat]])
    vector = np.hstack([fvec, gvec])
    solution = np.linalg.lstsq(matrix, vector)[0]
    expected = [0.0, 0.1, 0.1, 0.2, -10, 10]
    np.testing.assert_allclose(solution, expected, atol=1e-8)


@pytest.mark.parametrize("lagrange_type", ["standard", "augmented"])
def test_iterative_solver(Cmat, fvec, gvec, lagrange_type):
    class CallBack:
        def __init__(self, n: int):
            self.lagrange_multiplier = np.zeros(n)

    lag = LagrangeSolver(
        constraint_matrix=Cmat,
        constraint_vector=gvec,
        tolerance=1e-8,
        maxiters=100,
        lagrange_type=lagrange_type,
    )
    lag.update(lagrange_penalty=1e0)

    cb = CallBack(lag.constraint_matrix.shape[0])

    def residual(x):
        rprimal = fvec - matvec(x)
        rdual = lag.compute_residual_lag(rprimal, cb.lagrange_multiplier, x)
        return rdual, {}

    def increment(r, **kwargs):
        sol = kwargs["current_solution"]
        incr_u, incr_lag = lag.compute_increment_lag(matvec, None, r, sol)
        cb.lagrange_multiplier += incr_lag
        return incr_u

    solution = np.zeros_like(fvec)
    solver = NonLinearSolver(
        tolerance=1e-8,
        maxiters=100,
        allow_line_search=False,
        allow_acceleration=False,
    )
    solver.solve(solution, residual, increment)

    expected = [0.0, 0.1, 0.1, 0.2]
    np.testing.assert_allclose(solution, expected, atol=1e-8)

    expected = [-10, 10]
    np.testing.assert_allclose(cb.lagrange_multiplier, expected, 1e-8)
