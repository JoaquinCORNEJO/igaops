from typing import List, Any
from enum import Enum


class Constants:
    """
    Options available:
    - INFTY: 1e14
    - HUGE: 1e8
    - BIG: 1e4
    - SMALL: 1e-4
    - TINY: 1e-8
    - SAFEGUARD: 1e-14
    """

    INFTY = 1e14
    HUGE = 1e8
    BIG = 1e4
    SMALL = 1e-4
    TINY = 1e-8
    SAFEGUARD = 1e-14


class ParametricDirection(Enum):
    """
    Options available:
    - XI: first spatial parametric direction
    - ETA: second spatial parametric direction
    - NU: third spatial parametric direction
    - TAU: time parametric direction
    """

    XI = 0
    ETA = 1
    NU = 2
    TAU = 3
    ALL = -1


class BoundarySide(Enum):
    """
    Options available:
    - MIN: 0 in the parametric direction
    - MAX: 1 in the parametric direction
    - BOTH: 0 and 1
    """

    MIN = 0  # face 0 in [0,1]
    MAX = 1  # face 1 in [0,1]
    BOTH = -1


def validate_entry(value: str, reference: List[str]) -> Any:
    "Verify if value is in reference list."
    if not isinstance(value, str):
        raise TypeError(f"Value's type must be string, got {type(value)}")

    if not value in reference:
        raise ValueError(
            f"Invalid value. Got '{value}', but must be one of {reference}"
        )
    return value
