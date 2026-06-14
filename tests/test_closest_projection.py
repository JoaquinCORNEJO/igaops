from geomdl import NURBS, BSpline, operations
from typing import Tuple
import numpy as np
import pytest

from igaops.common import BoundarySide, ParametricDirection
from igaops.geometry import SinglePatch, GeometryOps


@pytest.fixture
def localization():
    # Commun localization for both examples
    return (ParametricDirection.XI, BoundarySide.MIN)


# First example


@pytest.fixture
def plate_with_hole():
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

    return SinglePatch.from_geomdl(
        surf, quadclass="gauss", quadtype="legendre"
    ).generate()


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_closest_point_projection_0(plate_with_hole, localization, method):
    X_target = np.array(
        [
            [np.sqrt(3) / 2, 1 / 2],
            [1 / 2, np.sqrt(3) / 2],
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [0.999, np.sqrt(1 - 0.999**2)],
        ]
    ).T

    pts, _ = plate_with_hole.closest_point_projection(
        X_target=X_target,
        localization=localization,
        method=method,
    )

    np.testing.assert_allclose(
        pts[1],
        [0.34108138, 0.65891862, 0.5, 0.031340338],
    )


# Second example


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
    P2 = [np.cos(alpha_end), np.sin(alpha_end), 0.0, 1.0]

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


def create_nurbs_patch(
    degree_u: int, degree_v: int, nbel_u: int, nbel_v: int, **geo_args
) -> NURBS.Surface:

    assert degree_u > 1 and degree_v > 0

    RINT = geo_args["radius_int"]
    REXT = geo_args["radius_ext"]
    THET = geo_args["angle_extension"]

    # Construction of the arc
    obj_arc = create_NURBS_arc(degree_v, nbel_v, alpha_ini=THET[0], alpha_end=THET[1])
    knotvector_v = obj_arc.knotvector
    ctrlpts_arc = obj_arc.ctrlpts
    weights_arc = obj_arc.weights

    # Construction of line
    knotvector_u, ctrlpts_line = make_BSPLINE_line(degree_u, nbel_u)
    ctrlpts_line = RINT + ctrlpts_line * (REXT - RINT)

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


@pytest.fixture
def quarter_annulus():
    DEGREE = 2
    NBEL = 2
    geo_args = {
        "radius_int": 1.5,
        "radius_ext": 2.0,
        "angle_extension": (0.0, np.pi / 2),
    }
    geometry = create_nurbs_patch(
        degree_u=DEGREE, degree_v=DEGREE, nbel_u=NBEL, nbel_v=NBEL, **geo_args
    )
    return SinglePatch.from_geomdl(
        geometry, quadclass="gauss", quadtype="legendre"
    ).generate()


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_closest_point_projection_1(quarter_annulus, localization, method):
    x0 = 1.499998499121685
    y0 = np.sqrt(1.5**2 - x0**2)
    X_target = np.array([[x0, y0, 0.0]]).T
    pts, _ = quarter_annulus.closest_point_projection(
        X_target=X_target,
        localization=localization,
        method=method,
    )
    np.testing.assert_allclose(pts[1], [0.001], rtol=1e-6)


# Third example


@pytest.fixture
def cube():

    DEGREE = 2
    NBEL = 2
    NBCTRLPS = DEGREE + NBEL

    # Set control points
    knotvector_u, ctrlpts_u = make_BSPLINE_line(DEGREE, NBEL)
    knotvector_v, ctrlpts_v = make_BSPLINE_line(DEGREE, NBEL)
    knotvector_w, ctrlpts_w = make_BSPLINE_line(DEGREE, NBEL)

    ctrlpts = [[xt, yt, zt] for zt in ctrlpts_w for xt in ctrlpts_u for yt in ctrlpts_v]

    # Create volume
    obj = BSpline.Volume()
    obj.degree_u = DEGREE
    obj.degree_v = DEGREE
    obj.degree_w = DEGREE
    obj.set_ctrlpts(ctrlpts, NBCTRLPS, NBCTRLPS, NBCTRLPS)
    obj.knotvector_u = knotvector_u
    obj.knotvector_v = knotvector_v
    obj.knotvector_w = knotvector_w

    return SinglePatch.from_geomdl(
        obj, quadclass="gauss", quadtype="legendre"
    ).generate()


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_closest_point_projection_2(cube, localization, method):
    X_target = np.array([[0.0, 0.5, 0.6], [0.0, 0.2, 0.4]]).T

    pts, _ = cube.closest_point_projection(
        X_target=X_target,
        localization=localization,
        method=method,
    )

    np.testing.assert_allclose(pts, X_target, rtol=1e-6)
