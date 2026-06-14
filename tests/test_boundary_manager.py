from geomdl import BSpline
import numpy as np
import pytest

from igaops.common import BoundarySide, ParametricDirection
from igaops.geometry import SinglePatch, GeometryOps
from igaops.operators import BsplineOperations

DEGREE_U = 1
DEGREE_V = 2
DEGREE_W = 3
NBEL = 1


@pytest.fixture
def patch():
    # Knot vectors
    knotvector_u = GeometryOps.make_open_knotvector(DEGREE_U, NBEL)
    ctrlpts_u = GeometryOps.evaluate_greville(DEGREE_U, knotvector_u)

    knotvector_v = GeometryOps.make_open_knotvector(DEGREE_V, NBEL)
    ctrlpts_v = GeometryOps.evaluate_greville(DEGREE_V, knotvector_v)

    knotvector_w = GeometryOps.make_open_knotvector(DEGREE_W, NBEL)
    ctrlpts_w = GeometryOps.evaluate_greville(DEGREE_W, knotvector_w)

    # Control points
    ctrlpts = [[xt, yt, zt] for zt in ctrlpts_w for xt in ctrlpts_u for yt in ctrlpts_v]

    # Create volume
    obj = BSpline.Volume()
    obj.degree_u = DEGREE_U
    obj.degree_v = DEGREE_V
    obj.degree_w = DEGREE_W

    obj.set_ctrlpts(
        ctrlpts,
        DEGREE_U + NBEL,
        DEGREE_V + NBEL,
        DEGREE_W + NBEL,
    )

    obj.knotvector_u = knotvector_u
    obj.knotvector_v = knotvector_v
    obj.knotvector_w = knotvector_w

    return SinglePatch.from_geomdl(
        obj,
        quadclass="gauss",
        quadtype="legendre",
    ).generate()


@pytest.mark.parametrize(
    "localization,quadrature_ids",
    [
        (
            (ParametricDirection.XI, BoundarySide.MIN),
            ("eta", "nu"),
        ),
        (
            (ParametricDirection.ETA, BoundarySide.MAX),
            ("xi", "nu"),
        ),
    ],
)
def test_boundary_data(patch, localization, quadrature_ids):
    quadrule_xi, quadrule_eta, quadrule_nu = patch.quadrule_list

    quadrules = {
        "xi": quadrule_xi,
        "eta": quadrule_eta,
        "nu": quadrule_nu,
    }

    indices = patch.boundary_manager.select(localization)

    new_ctrlpts = patch.ctrlpts[:, indices]

    bases = [quadrules[q].basis for q in quadrature_ids]

    expected_quadpts = BsplineOperations.eval_interpolation(
        bases,
        new_ctrlpts,
    )

    expected_jac = BsplineOperations.eval_jacobien(
        bases,
        new_ctrlpts,
    )

    surf = patch.get_data_on_boundary(localization=localization)

    np.testing.assert_allclose(expected_quadpts, surf.qp_phy)
    np.testing.assert_allclose(expected_jac, surf.jac)
