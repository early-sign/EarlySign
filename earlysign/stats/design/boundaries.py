"""Boundary value calculation for group sequential designs."""

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from earlysign.stats.design.config import DesignSpec
from earlysign.stats.design.types import SpendingFunction


class BoundaryCalculator:
    """Calculate critical boundary values for sequential designs.

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> boundaries = BoundaryCalculator.critical_values(spec)
    >>> 'z_efficacy' in boundaries
    True
    >>> len(boundaries['info_times'])
    3
    """

    @staticmethod
    def spending_function(
        func: SpendingFunction, t: np.ndarray, alpha: float, gamma: float = -4.0
    ) -> np.ndarray:
        """Calculate cumulative alpha spending at information times.

        Args:
            func: Spending function type
            t: Information times in (0, 1]
            alpha: Total alpha to spend
            gamma: HSD gamma parameter (only used for HSD function)

        Returns:
            Cumulative alpha spent at each information time

        Raises:
            ValueError: If spending function is unknown

        >>> t = np.array([0.5, 1.0])
        >>> spent = BoundaryCalculator.spending_function(
        ...     SpendingFunction.OBRIEN_FLEMING, t, 0.025
        ... )
        >>> len(spent)
        2
        >>> bool(0.04 < spent[-1] < 0.06)  # OBF spends more at final analysis
        True
        """
        if func == SpendingFunction.OBRIEN_FLEMING:
            # Lan-DeMets O'Brien-Fleming approximation
            z_alpha = norm.ppf(1 - alpha)
            result: np.ndarray = 2 * (1 - norm.cdf(z_alpha / np.sqrt(t)))
            return result
        elif func == SpendingFunction.POCOCK:
            # Pocock-like spending
            result = alpha * np.log(1 + (np.e - 1) * t)
            return result
        elif func == SpendingFunction.HSD:
            # Hwang-Shih-DeCani
            if abs(gamma) < 1e-12:
                return alpha * t
            result = alpha * (1 - np.exp(-gamma * t)) / (1 - np.exp(-gamma))
            return result
        else:
            raise ValueError(f"Unknown spending function: {func}")

    @staticmethod
    def critical_values(spec: DesignSpec) -> Dict[str, Any]:
        """Calculate critical values for efficacy and futility bounds.

        Args:
            spec: Design specification

        Returns:
            Dictionary containing:
                - info_times: Information times
                - cumulative_alpha: Cumulative alpha spent
                - z_efficacy: Efficacy Z-value thresholds
                - z_futility: Futility Z-value thresholds (if enabled)

        >>> from earlysign.stats.design.config import ProportionsDesignSpec
        >>> spec = ProportionsDesignSpec()
        >>> spec.sequential.n_analyses = 2
        >>> boundaries = BoundaryCalculator.critical_values(spec)
        >>> boundaries['z_efficacy'].shape[0]
        2
        """
        t = spec.resolved_info_times()
        k = len(t)

        # Efficacy spending
        alpha_total = (
            spec.test.alpha if spec.test.sided == "one" else spec.test.alpha / 2.0
        )
        cumulative_alpha = BoundaryCalculator.spending_function(
            spec.boundary.spending_function, t, alpha_total, spec.boundary.hsd_gamma
        )

        # Incremental alpha at each analysis
        alpha_increments = np.diff(np.concatenate([[0.0], cumulative_alpha]))

        # Critical Z-values (approximate, assuming independence)
        z_efficacy = norm.ppf(1 - alpha_increments)

        # Futility bounds (if enabled)
        z_futility: Optional[np.ndarray] = None
        if spec.boundary.futility_enabled:
            if spec.boundary.futility_z is not None:
                # Non-binding constant futility bound
                z_futility = np.full(k, spec.boundary.futility_z)
            elif spec.boundary.futility_spending is not None:
                # Spending function based futility
                beta = 1 - spec.test.power
                cumulative_beta = BoundaryCalculator.spending_function(
                    spec.boundary.futility_spending, t, beta, spec.boundary.hsd_gamma
                )
                beta_increments = np.diff(np.concatenate([[0.0], cumulative_beta]))
                z_futility = norm.ppf(beta_increments)
            else:
                # Binding futility proportional to efficacy (if not specified otherwise)
                z_futility = 0.5 * z_efficacy

        return {
            "info_times": t,
            "cumulative_alpha": cumulative_alpha,
            "z_efficacy": z_efficacy,
            "z_futility": z_futility,
        }
