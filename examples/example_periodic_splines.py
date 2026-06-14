from pathlib import Path
import os

from matplotlib import pyplot as plt
import numpy as np

from igaops.geometry import GeometryOps
from igaops.quadrature import StandardGauss

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

degree, nbelem = 3, 6
knotvector = GeometryOps.make_closed_knotvector(degree, nbelem)
quadrule = StandardGauss(degree, knotvector, quadtype="legendre")
knots = np.linspace(0, 1, 101)
basis = quadrule.eval_basis(knots)

# Figure
basis_idx = 3
fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i in range(quadrule.nbctrlpts):
    ax1.plot(knots, basis[0][:, i].toarray())
ax1.plot([], [], linewidth=1, color="k")
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\hat{b}_{A,\,p}(\xi)$")
ax1.set_ylim([0, 1])

ax1copy = ax1.twinx()
ax1copy.plot(quadrule.quadpts, quadrule.weights[0][basis_idx, :].toarray(), "ko")
ax1copy.grid(None)
ax1copy.set_ylim([0, 0.1])

for i in range(quadrule.nbctrlpts):
    ax2.plot(knots, basis[-1][:, i].toarray())
ax2.plot([], [], linewidth=1, color="k", label="B-Spline basis")
ax2.set_xlabel(r"$\xi$")
ax2.set_ylabel(r"$\hat{b}_{A,\,p}(\xi)$")

ax2copy = ax2.twinx()
ax2copy.plot(quadrule.quadpts, quadrule.weights[-1][basis_idx, :].toarray(), "ko")
ax2copy.grid(None)
ax2copy.set_ylim([-0.5, 0.5])

ctrlpts = np.array(
    [[0, 0], [1, 2], [2, 4], [5, 0.5], [4, 0.25], [3, 0], [0, 0], [1, 2], [2, 4]]
)
evalpts = basis[0] @ ctrlpts
ax3.plot(ctrlpts[:, 0], ctrlpts[:, 1])
ax3.plot(evalpts[:, 0], evalpts[:, 1], "--")
ax3.set_xlabel(r"$x_1$")
ax3.set_ylabel(r"$x_2$")

fig.tight_layout()
fig.savefig(FOLDER / "iga_periodic_basis", dpi=300)
