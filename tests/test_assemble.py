from typing import Tuple

from geomdl import NURBS, operations
import numpy as np
import pytest

from igaops.geometry import GeometryOps, SinglePatch
from igaops.operators import NurbsOperations


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


@pytest.fixture
def quarter_annulus():
    "Creates a quarter of a ring (or annulus) using NURBS"

    # Geometry properties
    degree_u = degree_v = 3
    nbel_u = nbel_v = 3
    Rint = 1.0
    Rext = 2.0

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

    return SinglePatch.from_geomdl(
        obj, quadclass="gauss", quadtype="legendre"
    ).generate()


@pytest.mark.parametrize("method", ["u_v", "gradu_gradv", "u_gradv", "gradu_v"])
def test_assembling(quarter_annulus, method):
    matvec = {
        "u_v": NurbsOperations.compute_mf_scalar_u_v,
        "gradu_gradv": NurbsOperations.compute_mf_scalar_gradu_gradv,
        "u_gradv": NurbsOperations.compute_mf_scalar_u_gradv,
        "gradu_v": NurbsOperations.compute_mf_scalar_gradu_v,
    }[method]

    assemble = {
        "u_v": NurbsOperations.assemble_scalar_u_v,
        "gradu_gradv": NurbsOperations.assemble_scalar_gradu_gradv,
        "u_gradv": NurbsOperations.assemble_scalar_u_gradv,
        "gradu_v": NurbsOperations.assemble_scalar_gradu_v,
    }[method]

    # Recover data from geometry
    quadrule_list = quarter_annulus.quadrule_list
    nurbs_weights = quarter_annulus.nurbs_weights

    rng = np.random.default_rng(0)
    v_in = rng.random(quarter_annulus.nbctrlpts_total)

    if method == "u_v":
        coefficients = quarter_annulus.det_jac
    elif method == "gradu_gradv":
        eye = np.ones((2, 2, quarter_annulus.nbquadpts_total))
        coefficients = eye * quarter_annulus.det_jac[np.newaxis, np.newaxis, :]
    elif method == "u_gradv" or method == "gradu_v":
        eye = np.ones((2, quarter_annulus.nbquadpts_total))
        coefficients = eye * quarter_annulus.det_jac[np.newaxis, :]
    else:
        raise ValueError

    # Use matrix-vector as reference
    v_refe = matvec(quadrule_list, coefficients, v_in, nurbs_weights=nurbs_weights)

    # Assemble matrix to test
    matrix = assemble(quadrule_list, coefficients, nurbs_weights=nurbs_weights)
    v_test = matrix @ v_in

    np.testing.assert_allclose(v_test, v_refe)
