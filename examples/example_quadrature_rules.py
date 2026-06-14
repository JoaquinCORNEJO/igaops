from pathlib import Path
import os

from matplotlib import pyplot as plt
import numpy as np

from igaops.quadrature import WeightedQuadrature
from igaops.geometry import GeometryOps

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)


degree, nbelem = 3, 4
knotvector = GeometryOps.make_open_knotvector(degree, nbelem)
quadrule = WeightedQuadrature(
    degree, knotvector, quadtype="2", position_rule="cellcenter"
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule.unique_kv, np.zeros_like(quadrule.unique_kv), marker="s", color="k")
weightsmatrix = quadrule.weights[0].toarray()
for i in range(quadrule.nbctrlpts):
    ax.plot(quadrule.quadpts, weightsmatrix[i])
fig.savefig(FOLDER / "weighted_quadrature_ex1_W0.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule.unique_kv, np.zeros_like(quadrule.unique_kv), marker="s", color="k")
weightsmatrix = quadrule.weights[-1].toarray()
for i in range(quadrule.nbctrlpts):
    ax.plot(quadrule.quadpts, weightsmatrix[i])
fig.savefig(FOLDER / "weighted_quadrature_ex1_W1.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule.unique_kv, np.zeros_like(quadrule.unique_kv), marker="s", color="k")
basismatrix = quadrule.basis[0].toarray().T
for i in range(quadrule.nbctrlpts):
    ax.plot(quadrule.quadpts, basismatrix[i])
fig.savefig(FOLDER / "weighted_quadrature_ex1_B0.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(quadrule.unique_kv, np.zeros_like(quadrule.unique_kv), marker="s", color="k")
basismatrix = quadrule.basis[-1].toarray().T
for i in range(quadrule.nbctrlpts):
    ax.plot(quadrule.quadpts, basismatrix[i])
fig.savefig(FOLDER / "weighted_quadrature_ex1_B1.png")
