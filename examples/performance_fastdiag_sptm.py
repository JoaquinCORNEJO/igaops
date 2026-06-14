import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr

from time import process_time
from pathlib import Path
import pandas as pd
import numpy as np

from igaops.quadrature import StandardGauss
from igaops.fastdiag import SingleFD
from igaops.geometry import GeometryOps

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)


def simulation(degree: int, nbel: int, ndim=2):
    NTIMES = 10
    NDOF = 1
    knotvector = GeometryOps.make_open_knotvector(degree, nbel)
    quadrule = StandardGauss(degree, knotvector, quadtype="legendre")
    quadrule_list = [quadrule] * ndim

    fd = SingleFD()
    table_dirichlet = np.ones((NDOF, ndim, 2), dtype=bool)
    fd.compute_space_eigendecomposition(quadrule_list, table_dirichlet)
    fd.compute_time_schurdecomposition(quadrule)

    rng = np.random.default_rng(0)
    v = rng.random(
        NDOF * fd.space_preconditioner.sp_nnz * fd.sptm_preconditioner.tm_nnz
    )
    fd.apply_spacetime_preconditioner(v)

    time_sum = 0.0
    for _ in range(NTIMES):
        start = process_time()
        fd.apply_spacetime_preconditioner(v)
        finish = process_time()
        time_sum += finish - start
    time_total = time_sum / NTIMES
    return time_total


results = []
degree_list = np.arange(2, 21).astype(int)
nbel_list = np.array([2**c for c in range(3, 8)]).astype(int)
for p in degree_list:
    for n in nbel_list:
        t = simulation(degree=p, nbel=n)
        results.append({"degree": p, "nbel": n, "time": t})

filename = FOLDER / "cpu_time_sptm_fastdiag.csv"
df = pd.DataFrame(results)
df.to_csv(filename, index=False)
