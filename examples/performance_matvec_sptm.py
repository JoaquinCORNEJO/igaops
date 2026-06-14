import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr

from typing import Literal
from time import process_time
from pathlib import Path
import pandas as pd
import numpy as np

from igaops.quadrature import StandardGauss, WeightedQuadrature
from igaops.operators import BsplineOperations
from igaops.geometry import GeometryOps

FILEPATH = Path(os.path.realpath(__file__)).parent
FOLDER = FILEPATH / "results"
if not FOLDER.exists():
    FOLDER.mkdir()


def simulation(degree: int, nbel: int, quadclass: Literal["gauss", "weighted"], ndim=2):
    NTIMES = 10
    knotvector = GeometryOps.make_open_knotvector(degree, nbel)
    generator = StandardGauss if quadclass == "gauss" else WeightedQuadrature
    quadtype = "legendre" if quadclass == "gauss" else "2"
    quadrule = generator(degree, knotvector, quadtype=quadtype)
    quadrule_list = [quadrule] * (ndim + 1)
    nbctrlpts_total = quadrule.nbctrlpts ** (ndim + 1)
    nbquadpts_total = quadrule.nbquadpts ** (ndim + 1)

    rng = np.random.default_rng(0)
    v = rng.random(nbctrlpts_total)

    coefficients_1 = np.ones(nbquadpts_total)
    coefficients_2 = np.ones((3, 3, nbquadpts_total))

    time_sum = 0.0
    for _ in range(NTIMES):
        start = process_time()
        BsplineOperations.compute_mf_scalar_u_v(
            quadrule_list,
            coefficients_1,
            v,
            time_ders=(0, 1),
            nurbs_weights=np.array([]),
        )
        BsplineOperations.compute_mf_scalar_gradu_gradv(
            quadrule_list,
            coefficients_2,
            v,
            time_ders=(0, 0),
            nurbs_weights=np.array([]),
        )
        finish = process_time()
        time_sum += finish - start
    time_total = time_sum / NTIMES
    return time_total


results = []
degree_list = np.arange(2, 21).astype(int)
nbel_list = np.array([2**c for c in range(3, 7)]).astype(int)
for n in nbel_list:
    for p in degree_list:
        t = simulation(degree=p, nbel=n, quadclass="weighted")
        results.append({"degree": p, "nbel": n, "time": t})

        filename = FOLDER / "cpu_time_sptm_matvec.csv"
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
