from typing import Union, Literal, Optional

from scipy.sparse.linalg import LinearOperator
from scipy import sparse as sp
import numpy as np

from igaops.common import validate_entry
from .lagrange.augmented import AugmentedLagrange
from .lagrange.standard import StandardLagrange

array_like = Union[np.ndarray, sp.csr_array, LinearOperator]


class LagrangeSolver:
    """
    Lagrange solver class: helper for computing the residual and increments for Lagrange methods.

    The kind of problems we are interested on is:
        f_int(u) = f, constrained to C u = g
    This nonlinear system is solved using a Newton solver with Lagrange approach.

    We only consider two cases:
        - Augmented Lagrangian method (ALM) in monoblock.
        - Standard Lagragian method using null space.

    Then, the system can be rewritten as:
        res_u = f - f_int(u) = 0, constrained to res_lam = g - C u = 0
    If not stated differently, all the variables are computed at iteration k.

    Notation:
        - T stands for the tangent matrix of res_u.
        - P stands for a good preconditioner for matrix T.
        - lam stands for the curent Lagrange multipliers.
        - U stands for the current solution.

    This class helps
        - to compute the residual that includes information of Lagrange multipliers.
        - to compute the increment of u for the updating in nonlinear solver.

    Standard method
    ---------------
    This method computes the null space of C, denote by Z, and solves a reduced system.
    The residual is updated as follows:
        res_u_update = res_u - C^T lam

    To compute the increment, it is necessary to follow:
    1. compute res_lam = g - C U
    2. solve C delta_up = res_lam using least-squares
    3. solve (Z^T T Z) y = Z^T (res_u_update - T delta_up)
    4. compute delta_ug = Z y
    5. set increment as delta_u* = delta_up + delta_ug
    6. new Lagrange multiplier is set after solving C^T lam = res_u_update - T delta_u*


    ALM method
    -----------
    In this approach, the residual is updated as follows:
        res_u_update = res_u - C^T lam + rho * C^T res_lam
    where res_lam = g - C U, as stated before.

    In this case, increments are compute in monoblock, i.e., M X = R
    where
        - M = [T + rho * C^T C, C^T; C, 0]
        - X = [delta_u*; delta_lam]
        - R = [res_u_updated; res_lam]
    In this case the Lagrange multiplier should also be updated every iteration.

    Parameters
    ----------
    constraint_matrix : array_like
        Matrix C in our formulation.
    constraint_vector : np.ndarray
        Vector g in our formulation.
    tolerance : float
        Linear solver tolerance.
    maxiters : int
        Linear solver number of iterations.
    lagrange_type : {"augmented", "standard"}
        Method chossen.
    verbose : bool
        If true, prints information of solvers.
    dual_constraint_matrix : array_like, optional
       Allows asymmetrical formulation in ALM solver.
    """

    registry = {
        "augmented": AugmentedLagrange,
        "standard": StandardLagrange,
    }

    def __new__(
        cls,
        constraint_matrix: array_like,
        constraint_vector: np.ndarray,
        tolerance: float,
        maxiters: int,
        lagrange_type: Literal["augmented", "standard"] = "augmented",
        verbose: bool = True,
        dual_constraint_matrix: Optional[array_like] = None,
    ) -> Union[AugmentedLagrange, StandardLagrange]:
        validate_entry(lagrange_type, list(cls.registry.keys()))
        subclass = cls.registry[lagrange_type]
        return subclass(
            constraint_matrix,
            constraint_vector,
            tolerance,
            maxiters,
            verbose,
            dual_constraint_matrix,
        )
