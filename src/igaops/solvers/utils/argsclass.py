from dataclasses import dataclass

import numpy as np

from igaops.common import Constants


@dataclass
class SolverArgs:
    """Configuration for solver."""

    tolerance: float
    maxiters: int

    def __post_init__(self):
        """Validate arguments after initialization."""
        if self.tolerance < Constants.SAFEGUARD or self.tolerance >= 1:
            # It should not be lees than the machine error (in practice around 1e-14)
            # and it could not be greater than 1 (an error of 100%)
            raise ValueError(
                f"Relative tolerance must be > 0 and < 1, got {self.tolerance}"
            )
        if self.maxiters <= 0:
            # It should a be a positve number of iterations
            raise ValueError(f"Max iterations must be > 0, got {self.maxiters}")


@dataclass
class ConvergenceArgs:
    """Configuration for convergence criteria."""

    relative_tolerance: float
    max_iterations: int
    absolute_tolerance: float = Constants.SAFEGUARD

    def __post_init__(self):
        """Validate criteria after initialization."""
        if self.relative_tolerance <= 0 or self.relative_tolerance >= 1:
            raise ValueError(
                f"Relative tolerance must be > 0 and < 1, got {self.relative_tolerance}"
            )
        if self.max_iterations <= 0:
            raise ValueError(f"Max iterations must be > 0, got {self.max_iterations}")
        if self.absolute_tolerance < 0:
            raise ValueError(
                f"Absolute tolerance must be >= 0, got {self.absolute_tolerance}"
            )


@dataclass
class InnerToleranceArgs:
    """Arguments for inner tolerance computation."""

    initial_tolerance: float = 0.5
    static: float = 0.1
    coefficient: float = 0.9
    exponential: float = 1.5
    default: float = Constants.TINY

    def __post_init__(self):
        """Validate arguments after initialization."""
        for key, value in self.__dict__.items():
            if not np.isscalar(value):
                raise ValueError(f"{key} must be a scalar value")
            if key == "default" and np.real(value) >= 1.0:
                raise ValueError(f"default must be < 1.0, got {value}")
            if key == "initial_tolerance" and np.real(value) >= 1.0:
                raise ValueError(f"Initial value must be < 1.0, got {value}")
            if key == "static" and np.real(value) >= 1.0:
                raise ValueError(f"Static value must be < 1.0, got {value}")
