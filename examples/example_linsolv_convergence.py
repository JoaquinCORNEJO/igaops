from typing import Literal, Tuple
from pathlib import Path
import os

from scipy.sparse.linalg import LinearOperator
from scipy.sparse import linalg as scsplin
from geomdl import NURBS, operations
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from igaops.geometry import SinglePatch, GeometryOps
from igaops.solvers import LinearSolver, RandomPreconditioner
from igaops.fastdiag import SingleFD

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)


def make_BSPLINE_line(degree: int, nbel: int) -> Tuple[np.ndarray, np.ndarray]:
    knotvector = GeometryOps.make_open_knotvector(degree, nbel)
    nbctrlpts = len(knotvector) - degree - 1
    ctrlpts = np.array(
        [
            sum(knotvector[i + j + 1] for j in range(degree)) / degree
            for i in range(0, nbctrlpts)
        ]
    )
    return knotvector, ctrlpts


def create_NURBS_arc(
    degree: int, nbel: int, alpha_ini: float, alpha_end: float
) -> NURBS.Curve:

    assert degree > 1
    # Arc parameters
    beta = 0.5 * (alpha_ini + alpha_end)
    w = np.cos((alpha_end - alpha_ini) / 2)

    P0 = [np.cos(alpha_ini), np.sin(alpha_ini), 0.0, 1.0]
    P1 = [np.cos(beta), np.sin(beta), 0.0, w]
    P2 = [np.cos(alpha_end), np.sin(alpha_end), 0, 1.0]

    # Create the vanilla circle using NURBS
    obj = NURBS.Curve()
    obj.degree = 2
    obj.ctrlptsw = [P0, P1, P2]
    obj.knotvector = [0, 0, 0, 1, 1, 1]

    # Add degree elevation
    if degree > 2:
        operations.degree_operations(obj, [degree - 2])

    # Add knot refinement
    for knot in np.linspace(0.0, 1.0, nbel + 1)[1:-1]:
        operations.insert_knot(obj, [knot], [1])

    return obj


def create_nurbs_quarter_annulus(
    degree_u: int, degree_v: int, nbel_u: int, nbel_v: int, Rint: float, Rext: float
) -> NURBS.Surface:
    "Creates a quarter of a ring (or annulus) using NURBS"

    # Construction of the arc
    obj_arc = create_NURBS_arc(degree_v, nbel_v, 0.0, np.pi / 2)
    knotvector_v = obj_arc.knotvector
    ctrlpts_arc = obj_arc.ctrlpts
    weights_arc = obj_arc.weights

    # Construction of line
    knotvector_u, ctrlpts_line = make_BSPLINE_line(degree_u, nbel_u)
    ctrlpts_line = Rint + ctrlpts_line * Rext - ctrlpts_line * Rint

    # Construction of annulus sector
    ctrlpts = [
        [x_line * x_arc * w_arc, x_line * y_arc * w_arc, 0.0, w_arc]
        for x_line in ctrlpts_line
        for (x_arc, y_arc, _), w_arc in zip(ctrlpts_arc, weights_arc)
    ]

    # Create surface
    obj = NURBS.Surface()
    obj.degree_u = degree_u
    obj.degree_v = degree_v
    obj.set_ctrlpts(ctrlpts, len(ctrlpts_line), len(ctrlpts_arc))
    obj.knotvector_u = knotvector_u
    obj.knotvector_v = knotvector_v

    return obj


def simulate(
    degree: int,
    nbel: int,
    quadclass: Literal["gauss", "weighted"],
    quadtype: Literal["legendre", "1", "2"],
    preconditioner: Literal["wp", "jm", "fd", "ilu", "rand"] = "jm",
    linear_type: Literal["gmres"] = "gmres",
):
    RINT = 1.0
    REXT = 4.0
    geometry = create_nurbs_quarter_annulus(degree, degree, nbel, nbel, RINT, REXT)
    patch = SinglePatch.from_geomdl(
        geometry, quadclass=quadclass, quadtype=quadtype
    ).generate()

    def matvec(x):
        return patch.operator_engine.compute_mf_scalar_u_v(
            patch.quadrule_list,
            patch.det_jac,
            x,
            nurbs_weights=patch.nurbs_weights,
        )

    linearsolver = LinearSolver(
        tolerance=1e-12,
        maxiters=150,
        linear_type=linear_type,
    )

    Precond = None
    if preconditioner == "fd" or preconditioner == "jm":
        fd = SingleFD()
        fd.compute_space_eigendecomposition(
            patch.quadrule_list,
            np.zeros((1, patch.ndim, 2)),
        )
        if preconditioner == "jm":
            mc = float(np.mean(patch.det_jac))
            fd.add_scalar_space_time_correctors(mass_corrector=[mc])
        fd.update_space_eigenvalues(scalar_coefs=[1, 0])
        Precond = fd.apply_spatial_preconditioner

    elif preconditioner == "ilu":
        matrix = matvec(np.eye(patch.nbctrlpts_total))
        spilu = scsplin.spilu(matrix)
        Precond = lambda x: spilu.solve(x)

    elif preconditioner == "rand":
        # TODO: study randomized preconditioner
        MU = 1e-8

        def apply_Sfunc(x: np.ndarray) -> np.ndarray:
            t = matvec(x)
            return t - MU * x

        # Define random preconditioner
        nr = patch.nbctrlpts_total
        apply_S = LinearOperator(
            dtype=float, shape=(nr, nr), matvec=apply_Sfunc, rmatvec=apply_Sfunc
        )
        randprecond = RandomPreconditioner(
            S=apply_S,
            mu=MU,
            random_type="spd",  # "general" or "spd"?
            rank=1500,
            small_threshold=0,
        )
        randprecond.build()
        Precond = lambda x: randprecond.apply(x)

    rng = np.random.default_rng(0)
    noise = rng.random(patch.nbctrlpts_total)
    output = linearsolver.solve(
        Afun=matvec,
        b=noise,
        Pfun=Precond,
    )
    return output


# Run simulation
KEYS = ["wp", "fd", "ilu", "jm", "rand"]
DEGREE, NBEL = 6, 40
data = {}
for key in KEYS:
    output = simulate(
        DEGREE,
        NBEL,
        quadclass="weighted",
        quadtype="2",
        preconditioner=key,
    )
    data[key] = output.residual.tolist()
df = pd.DataFrame.from_dict(data, orient="index")
df = df.transpose()
df.to_csv(FOLDER / "linear_solver.csv")

# Post processing
MARKERLIST = ["o", "v", "s", "X", "+", "p", "*"]
COLORLIST = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]
labels = {
    "wp": "w.o. preconditioner",
    "fd": "Standard approach",
    "ilu": "Incomplete LU",
    "rand": "Random SVD",
    "jm": "Our approach",
}
df = pd.read_csv(FOLDER / "linear_solver.csv", index_col=0)
results_list = [df[key].dropna().tolist() for key in KEYS]

fig, ax = plt.subplots(figsize=(4, 4))
for j, label in enumerate([labels[ky] for ky in KEYS]):

    ax.semilogy(
        results_list[j],
        label=label,
        markevery=5,
        marker=MARKERLIST[j],
        color=COLORLIST[j],
    )

ax.legend(ncol=2, bbox_to_anchor=(0.45, 1.5), loc="upper center")
ax.set_ylabel("Relative residue")
ax.set_xlabel("Number of iterations (GMRES)")
ax.set_ylim(bottom=1e-12, top=1e1)
ax.set_xlim(left=0, right=100)
fig.tight_layout()
fig.savefig(FOLDER / "linear_solver.pdf", bbox_inches="tight")
