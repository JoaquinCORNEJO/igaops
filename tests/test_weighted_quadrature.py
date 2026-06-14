import numpy as np
import pytest

from igaops.quadrature import WeightedQuadrature, StandardGauss
from igaops.geometry import GeometryOps


def relative_error(a, b):
    diff = b - a
    return np.linalg.norm(diff.toarray()) / np.linalg.norm(b.toarray())


@pytest.mark.parametrize("position_rule", ["endpoint", "cellcenter"])
@pytest.mark.parametrize("quadtype", ["1", "2"])
@pytest.mark.parametrize("var_name", ["I00", "I01", "I10", "I11"])
def test_weighted_quadrature_vs_gauss(position_rule, quadtype, var_name):

    nbel_list = [2**i for i in range(2, 5)]

    error_list = []
    for degree in range(2, 5):
        for nbel in nbel_list:

            knotvector = GeometryOps.make_open_knotvector(degree, nbel)

            # Weighted quadrature
            wq = WeightedQuadrature(
                degree,
                knotvector,
                quadtype=quadtype,
                position_rule=position_rule,
            )

            B0_wq, B1_wq = wq.basis
            W00_wq, W01_wq, W10_wq, W11_wq = wq.weights

            # Gauss quadrature
            gs = StandardGauss(degree, knotvector, quadtype="legendre")

            B0_gs, B1_gs = gs.basis
            W00_gs, W01_gs, W10_gs, W11_gs = gs.weights

            if var_name == "I00":
                var_ref = W00_gs @ B0_gs
                var_app = W00_wq @ B0_wq
            elif var_name == "I01":
                var_ref = W01_gs @ B1_gs
                var_app = W01_wq @ B1_wq
            elif var_name == "I10":
                var_ref = W10_gs @ B0_gs
                var_app = W10_wq @ B0_wq
            elif var_name == "I11":
                var_ref = W11_gs @ B1_gs
                var_app = W11_wq @ B1_wq
            else:
                raise ValueError("Wrong variable name")

            error_list.append(relative_error(var_app, var_ref))

    np.testing.assert_array_less(
        error_list,
        1e-8,
        err_msg=f"max: {max(error_list):.2e}, min: {min(error_list):.2e}",
    )
