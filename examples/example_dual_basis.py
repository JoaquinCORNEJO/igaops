from pathlib import Path
import os

from matplotlib import pyplot as plt
import scipy.sparse as sp
import numpy as np

from igaops.quadrature import StandardGauss
from igaops.geometry import GeometryOps

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)


def create_quadrule(degree: int, nbelem: int, r: int):
    knotvector = GeometryOps.make_open_knotvector(degree, nbelem)
    quadrule = StandardGauss(degree, knotvector, quadtype="legendre", nbptsperel=r)
    return quadrule


def make_dual_basis(
    degree: int, nbel_sl: int, nbel_ms: int, r_sl: int, compute_exact: bool = False
):
    # Create quadratures
    quadrule_sl = create_quadrule(degree, nbel_sl, r_sl)
    quadrule_ms = create_quadrule(degree, nbel_ms, r_sl)

    # Compute Gramm matrix with slave basis
    G_lam_sl = sp.csr_array(quadrule_sl.weights[0] @ quadrule_sl.basis[0]).toarray()

    if not compute_exact:
        basis_sl = quadrule_sl.basis[0].toarray().T
        Lambda = np.linalg.lstsq(G_lam_sl, basis_sl)[0]
        # NOTE: in this example the quadrature points of sl coincides with the master
        # Otherwise a projection is necessary
        basis_ms = quadrule_ms.eval_basis(quadrule_sl.quadpts)[0].toarray()
        G_lam_ms = Lambda @ np.diag(quadrule_sl.parametric_weights) @ basis_ms
        return quadrule_sl, G_lam_sl, G_lam_ms

    # Integrate with more elements
    new_knots = np.unique(np.hstack([quadrule_ms.unique_kv, quadrule_sl.unique_kv]))
    new_points, new_weights = quadrule_sl.interpolate_points_and_weights(new_knots)

    basis_sl = quadrule_sl.eval_basis(new_points)[0].toarray().T
    Lambda = np.linalg.lstsq(G_lam_sl, basis_sl)[0]
    basis_ms = quadrule_ms.eval_basis(new_points)[0].toarray()
    G_lam_ms = Lambda @ np.diag(new_weights) @ basis_ms

    return quadrule_sl, G_lam_sl, G_lam_ms


# Create reference
DEGREE, NBEL1, NBEL2 = 4, 4, 3
NBQUAD = 10
QUAD_SL, G_LAM_SL, G_LAM_MS = make_dual_basis(
    DEGREE, NBEL1, NBEL2, NBQUAD, compute_exact=True
)

# Plot
knots = np.linspace(0, 1, 101)
Lambda_to_plot = np.linalg.solve(G_LAM_SL, QUAD_SL.eval_basis(knots)[0].toarray().T)

fig, [ax1, ax2] = plt.subplots(figsize=(8, 5), ncols=2)
for i in range(Lambda_to_plot.shape[0]):
    ax1.plot(knots, Lambda_to_plot[i])
ax1.set_xlim(0, 1)
ax1.set_ylim(-20, 40)

# Plot
knots = np.linspace(0, 1, 101)
basis_to_plot = (QUAD_SL.eval_basis(knots)[0] @ G_LAM_MS).T

for i in range(basis_to_plot.shape[0]):
    ax2.plot(knots, basis_to_plot[i])
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.2, 1.2)

fig.tight_layout()
fig.savefig(FOLDER / "dual_mortar_basis.png")

for nq in range(DEGREE + 1, NBQUAD + 1):
    quadrule_sl, G_lam_sl, G_lam_ms = make_dual_basis(
        DEGREE, NBEL1, NBEL2, nq, compute_exact=False
    )

    err_sl = np.linalg.norm(G_lam_sl - G_LAM_SL)
    err_ms = np.linalg.norm(G_lam_ms - G_LAM_MS)

    print(
        f"number of quadpts: {nq}, "
        f"error_slave: {err_sl:.2e}, "
        f"error master: {err_ms:.2e}, "
    )
