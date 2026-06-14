import numpy as np
import pytest

from igaops.geometry import SinglePatch
from igaops.solvers import LinearSolver
from igaops.fastdiag import SingleFD
from geomdl import NURBS


@pytest.fixture
def patch():
    L = 4
    P1 = np.sqrt(2) - 1
    P2 = np.sqrt(2) / 2
    W = (np.sqrt(2) + 2) / 4

    ctrlpts = [
        [1.0, 0.0, 0.0, 1.0],
        [W, P1 * W, 0.0, W],
        [P2 * W, P2 * W, 0.0, W],
        [P1 * W, W, 0.0, W],
        [0.0, 1.0, 0.0, 1.0],
        [L, 0, 0, 1],
        [L, L / 2, 0, 1],
        [L, L, 0, 1],
        [L / 2, L, 0, 1],
        [0, L, 0, 1],
    ]

    surf = NURBS.Surface()
    surf.degree_u = 1
    surf.degree_v = 2

    surf.set_ctrlpts(ctrlpts, 2, 5)

    surf.knotvector_u = [0.0, 0.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]

    return SinglePatch.from_geomdl(surf, quadclass="weighted", quadtype="2").generate()


def test_matvec_operator(patch):
    def matvec(x):
        return patch.operator_engine.compute_mf_scalar_u_v(
            patch.quadrule_list,
            patch.det_jac,
            x,
            nurbs_weights=patch.nurbs_weights,
        )

    x = np.random.default_rng(0).random(patch.nbctrlpts_total)
    y = matvec(x)

    assert y.shape == x.shape
    assert np.isfinite(y).all()


@pytest.mark.parametrize("linear_type", ["cg", "bicgstab", "gcrodr", "gmres"])
def test_single_vs_block_solver_consistency(patch, linear_type):

    # matvec
    def matvec(x):
        return patch.operator_engine.compute_mf_scalar_u_v(
            patch.quadrule_list,
            patch.det_jac,
            x,
            nurbs_weights=patch.nurbs_weights,
        )

    # preconditioner
    fd = SingleFD()
    fd.compute_space_eigendecomposition(
        patch.quadrule_list,
        np.zeros((1, patch.ndim, 2)),
    )
    fd.update_space_eigenvalues(scalar_coefs=[1, 0])

    precond = fd.apply_spatial_preconditioner

    # RHS
    N = 5
    rng = np.random.default_rng(0)
    noise = rng.random((patch.nbctrlpts_total, N))

    solver = LinearSolver(
        tolerance=1e-8,
        maxiters=100,
        linear_type=linear_type,
    )

    # single RHS solves
    sol_single = np.zeros_like(noise)
    for i in range(N):
        sol_single[:, i] = solver.solve(
            Afun=matvec,
            b=noise[:, i],
            Pfun=precond,
            recycled_size=5,
        ).solution

    # block solve
    solver_block = LinearSolver(
        tolerance=1e-8,
        maxiters=100,
        linear_type="block_cg",
    )

    sol_block = solver_block.solve(
        Afun=matvec,
        b=noise,
        Pfun=precond,
    ).solution

    np.testing.assert_allclose(sol_single, sol_block, rtol=1e-6, atol=1e-6)
