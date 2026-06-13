from typing import Optional, Literal, Dict
import logging

import numpy as np

from igaops.common import Constants, validate_entry
from .argsclass import InnerToleranceArgs

logger = logging.getLogger(__name__)


class InnerToleranceSetter:
    """
    Manages inner tolerance computation for iterative solvers.

    Supports four tolerance strategies:
    - exact picard: Fixed tolerance at default value
    - exact newton: Fixed tolerance at default value
    - inexact picard: Static threshold-based tolerance
    - inexact newton: Adaptive tolerance based on residual reduction

    Parameters
    ----------
    inner_tolerance_type : str
        Strategy for tolerance computation.
    inner_tolerance_args : Dict[str, float]
        Dictionary of tolerance parameters (must include 'default').

    Raises
    ------
    AssertionError
        If tolerance type is invalid or args missing 'default'.
    """

    _VALID_TYPES = [
        "exact_picard",
        "exact_newton",
        "inexact_picard",
        "inexact_newton",
    ]

    def __init__(
        self,
        inner_tolerance_type: Literal[
            "exact_picard",
            "exact_newton",
            "inexact_picard",
            "inexact_newton",
        ],
        inner_tolerance_args: Dict[str, float],
    ):
        self._tolerance_type = validate_entry(inner_tolerance_type, self._VALID_TYPES)
        self._config = self._initialize_config(inner_tolerance_args)

    @property
    def tolerance_type(self) -> str:
        """Get current tolerance type."""
        return self._tolerance_type

    @property
    def config(self) -> InnerToleranceArgs:
        """Get current tolerance configuration."""
        return self._config

    def update(
        self,
        inner_tolerance_type: Optional[str] = None,
        **kwargs: float,
    ):
        """
        Update tolerance configuration.

        Parameters
        ----------
        inner_tolerance_type : str, optional
            New tolerance type.
        **kwargs : dict
            Additional arguments to update (e.g., default, coefficient).
        """
        if inner_tolerance_type is not None:
            self._tolerance_type = validate_entry(
                inner_tolerance_type, self._VALID_TYPES
            )

        for key, value in kwargs.items():
            if key in self.config.__dict__:
                if not np.isscalar(value):
                    raise ValueError(f"{key} must be a scalar value")
                if key == "default" and np.real(value) >= 1.0:
                    raise ValueError(f"default must be < 1.0, got {value}")
                setattr(self.config, key, float(np.real(value)))
            else:
                logger.warning(f"Unknown argument '{key}' ignored")

    def compute(
        self,
        norm_residual_new: Optional[float] = None,
        norm_residual_old: Optional[float] = None,
        inner_tolerance_old: Optional[float] = None,
    ) -> float:
        """
        Compute inner tolerance based on current strategy.

        Parameters
        ----------
        norm_residual_new : float, optional
            Current iteration residual norm.
        norm_residual_old : float, optional
            Previous iteration residual norm.
        inner_tolerance_old : float, optional
            Previous inner tolerance.

        Returns
        -------
        float
            Computed inner tolerance value
        """
        if self.tolerance_type in {"exact_picard", "exact_newton"}:
            return self._compute_exact()
        if self.tolerance_type == "inexact_picard":
            return self._compute_inexact_picard()
        if self.tolerance_type == "inexact_newton":
            return self._compute_inexact_newton(
                norm_residual_new, norm_residual_old, inner_tolerance_old
            )
        raise NotImplementedError(f"Unknown tolerance type: {self.tolerance_type}")

    def _initialize_config(self, config_params: Dict[str, float]) -> InnerToleranceArgs:
        """Initialize tolerance arguments from dictionary."""
        if not isinstance(config_params, dict):
            raise TypeError(f"Arguments must be dict, got {type(config_params)}")
        if "default" not in config_params:
            raise ValueError("Arguments must include 'default' value")

        return InnerToleranceArgs(**config_params)

    def _compute_exact(self) -> float:
        """Compute tolerance for exact methods (fixed at default)."""
        tolerance = self.config.default
        logger.debug(f"Exact tolerance: {tolerance:.2e}")
        return tolerance

    def _compute_inexact_picard(self) -> float:
        """Compute tolerance for inexact Picard method."""
        tolerance = max(
            self.config.default, min(self.config.initial_tolerance, self.config.static)
        )
        logger.debug(f"Inexact Picard tolerance: {tolerance:.2e}")
        return tolerance

    def _compute_inexact_newton(
        self,
        norm_residual_new: Optional[float],
        norm_residual_old: Optional[float],
        inner_tolerance_old: Optional[float],
    ) -> float:
        """
        Compute tolerance for inexact Newton method using Eisenstat-Walker.

        Uses adaptive strategy based on residual reduction rate.
        """
        threshold = self.config.initial_tolerance

        if (
            isinstance(norm_residual_new, (float, np.floating))
            and isinstance(norm_residual_old, (float, np.floating))
            and isinstance(inner_tolerance_old, (float, np.floating))
        ):
            gamma = self.config.coefficient
            omega = self.config.exponential

            ratio = norm_residual_new / max(norm_residual_old, Constants.TINY)
            eps_choice1 = gamma * np.power(ratio, omega)
            eps_choice2 = gamma * np.power(inner_tolerance_old, omega)

            # Use safeguarded choice
            if eps_choice2 > 0.1:
                threshold = min(eps_choice1, eps_choice2)
            else:
                threshold = eps_choice1

        tolerance = np.clip(
            threshold, self.config.default, self.config.initial_tolerance
        )
        logger.debug(f"Inexact Newton tolerance: {tolerance:.2e}")
        return tolerance
