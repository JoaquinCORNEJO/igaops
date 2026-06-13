from typing import Sequence, Tuple, List

from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_quadrature_layout(
    knot_vector_test: Sequence[float],
    knot_vector_target: Sequence[float],
    degree_test: int,
    degree_target: int,
) -> np.ndarray:
    """
    Compute the S4 constraint matrix for quadrature point layout.

    Parameters
    ----------
    knot_vector_test : array-like
        Open knot vector of the test space (includes repeated boundary knots).
    knot_vector_target : array-like
        Open knot vector of the target (trial) space.
    degree_test : int
        Polynomial degree of the test space.
    degree_target : int
        Polynomial degree of the target space.

    Returns
    -------
    S4 : np.ndarray, shape (n_el, n_el)
        Upper-triangular matrix of cumulative quadrature-point lower bounds.
        S4[r, s]  (0-indexed)  is the minimum total number of quadrature
        points required in elements r+1 through s+1 (1-indexed in the paper).
        The diagonal S4[e, e] is the per-element lower bound.
        Off-diagonal entries S4[r, s] with r < s are multi-element constraints.

    Notes
    -----
    This implements exactly steps S1-S4 of Figure 3 in the paper:

    S1  For each (test i, target j) pair, find the element-index interval
        (r, s) of their support intersection.  Single-point intersections
        (at a breakpoint only) are excluded — they have zero measure.

    S2  Accumulate multiplicities mu[(i, r, s)]: for each test function i
        count how many target functions j share exactly the interval (r, s).

    S3  For each (i, r, s), compute the cumulative count
            mu_bar(i, r, s) = sum_{k=r}^{s} sum_{l=k}^{s}  mu(i, k, l)
        which equals the minimum number of points needed in elements r..s
        to satisfy all constraints arising from test function i.

    S4  S4[r, s] = max over all test functions i of  mu_bar(i, r, s).
    """
    Xi_hat = np.asarray(knot_vector_test, dtype=float)
    Xi_bar = np.asarray(knot_vector_target, dtype=float)

    # Get unique values of knotvector
    # NOTE: it should be the same for test and target
    Delta = np.unique(knot_vector_test)
    n_el = len(Delta) - 1

    # Step 1
    supp_test = _element_supports(Xi_hat, degree_test, Delta)
    supp_target = _element_supports(Xi_bar, degree_target, Delta)
    S1 = _build_S1(supp_test, supp_target)

    # Step 2
    mu = _build_S2(S1, n_el)

    # Steps 3 and 4
    S4 = np.zeros((n_el, n_el), dtype=int)

    for mu_i in mu:
        B = _compute_mu_bar_table(mu_i)
        np.maximum(S4, B, out=S4)

    return S4.astype(int)


# ---------------------------------------------------------------------------
# S1 — element supports
# ---------------------------------------------------------------------------


def _element_supports(
    knot_vector: np.ndarray,
    degree: int,
    Delta: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Return the set of 0-indexed element indices covered by each B-spline.

    A B-spline N_{i,p} has support [xi_i, xi_{i+p+1}].  We map this to the
    elements it covers using the breakpoint sequence.  An element is covered
    only if the overlap has positive measure (not just a single point).
    """
    n_basis = len(knot_vector) - degree - 1
    supports = []

    for i in range(n_basis):
        a = knot_vector[i]
        b = knot_vector[i + degree + 1]
        elems = []
        for e in range(len(Delta) - 1):
            lo = max(a, Delta[e])
            hi = min(b, Delta[e + 1])
            if hi > lo:  # positive measure — strict inequality
                elems.append(e)
        supports.append((min(elems), max(elems)))

    return supports


def _build_S1(
    test_supports: List[Tuple[int, int]], target_supports: List[Tuple[int, int]]
) -> List[List[Tuple[int, int]]]:
    "Compute overlapping of functions using test and target supports"
    S1 = []
    for a, b in test_supports:
        row = []
        for c, d in target_supports:
            r = max(a, c)
            s = min(b, d)
            if r <= s:
                row.append((r, s))
        S1.append(row)
    return S1


# ---------------------------------------------------------------------------
# S2 — multiplicity accumulation
# ---------------------------------------------------------------------------


def _build_S2(S1, n_el) -> np.ndarray:
    """
    For each test function i, count how many target functions j have a
    support intersection with i equal to exactly the interval (r, s).

    The interval (r, s) is represented as (min_elem, max_elem) of the
    intersection set.  Target functions whose intersection with test i is
    not a contiguous block of elements are also assigned the (min, max)
    interval — the S3 sum then naturally handles their contribution through
    the sub-interval accumulation.
    """
    mu = np.zeros((len(S1), n_el, n_el), dtype=int)

    for i, row in enumerate(S1):
        for r, s in row:
            mu[i, r, s] += 1
    return mu


# ---------------------------------------------------------------------------
# S3 — cumulative count for one test function and one interval
# ---------------------------------------------------------------------------


def _compute_mu_bar_table(M: np.ndarray):
    """
    Precompute μ̄(r,s) for all r<=s.

    μ̄(r,s) is the sum of the upper-triangular part of the submatrix
    M[r:s+1, r:s+1]. The result is returned as an upper-triangular matrix
    B where B[r,s] = μ̄(r,s].
    """

    n = M.shape[0]

    col_cs = M.cumsum(axis=0)

    B = np.zeros_like(M).astype(int)

    for r in range(n):
        B[r, r] = M[r, r]

        running = B[r, r]

        for s in range(r + 1, n):
            colsum = col_cs[s, s]
            if r > 0:
                colsum -= col_cs[r - 1, s]

            running += colsum
            B[r, s] = running

    return B


# ---------------------------------------------------------------------------
# S5 — Greedy algorithm
# ---------------------------------------------------------------------------


def convert_to_greedy_input(S4: np.ndarray):
    """
    Convert output from ``compute_quadrature_layout`` (stage S4) to inputs for ``greedy_min_norm1``.
    """
    n_el = S4.shape[0]
    lb = np.diag(S4)

    rows_A, rows_b = [], []
    for r in range(n_el):
        for s in range(r + 1, n_el):  # off-diagonal only
            row = np.zeros(n_el, dtype=int)
            row[r : s + 1] = 1
            rows_A.append(row)
            rows_b.append(S4[r, s])

    A = np.array(rows_A, dtype=int)
    b = np.array(rows_b, dtype=int)
    return lb, A, b


def greedy_min_norm1(lb: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Closed-form optimal solution to:
        min  sum(x)
        s.t. x >= lb
        A @ x >= b    (A is 0/1, lb and b are positive integers)
        x integer

    It is a wraper to Scipy functions.

    Parameters
    ----------
    lb : (n,)   individual lower bounds
    A  : (m, n) 0/1 constraint matrix
    b  : (m,)   right-hand side

    Returns
    -------
    x : (n,) optimal integer solution
    """
    lb = np.asarray(lb, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Fast verification (since milp could be expensive)
    if np.all(A @ lb - b >= 0):
        return np.round(lb).astype(int)

    # Wrapper to milp
    result = milp(
        c=np.ones(lb.shape[0]),
        constraints=LinearConstraint(A, lb=b, ub=np.inf),
        integrality=np.ones(lb.shape[0]),
        bounds=Bounds(lb=lb, ub=np.inf),
    )

    if result.success:
        return np.round(np.asarray(result.x)).astype(int)
    raise ValueError(f"Solver failed {result.message}")
