import numpy as np
import pytest

from igaops.solvers import NonLinearSolver


def mat(x):
    return np.array(
        [
            [0.5 + x[1], 0.2, 0.1],
            [0.2, 1.0 + 2 * x[0], 0.0],
            [0.1, 0.0, 2.0 + x[2]],
        ]
    )


def compute_residual(x):
    b = np.array([2.0, 1.0, 0.5])
    return b - mat(x) @ x, {}


def solve_linearization_picard(res, **args):
    x = args.get("current_solution", np.zeros(3))
    return np.linalg.solve(mat(x), res)


@pytest.mark.parametrize("allow_acceleration", [False, True])
def test_nonlinear_solver(allow_acceleration):
    solver = NonLinearSolver(
        maxiters=20,
        tolerance=1e-9,
        allow_acceleration=False,
        allow_line_search=False,
    )

    # update only acceleration flag as in your script
    solver.update(allow_acceleration=allow_acceleration)

    solution = np.array([0.0, 0.0, 0.0])

    output = solver.solve(
        solution,
        compute_residual,
        solve_linearization_picard,
    )
    expected = np.array([3.75681868, 0.02920447, 0.06033869])
    np.testing.assert_allclose(solution, expected, rtol=1e-6, atol=1e-6)

    nb_iterations = len(output.nonlinear_time)
    expected = 11 if allow_acceleration else 21
    np.testing.assert_array_less(nb_iterations, expected)
