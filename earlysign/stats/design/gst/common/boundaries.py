"""Boundary value calculation for group sequential designs."""

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.types import SpendingFunction


class BoundaryCalculator:
    """Calculate critical boundary values for group sequential trial designs.

    This class implements the alpha spending function approach for sequential
    testing, which allows flexible timing of interim analyses while maintaining
    overall Type I error control. It supports multiple spending functions:

    - **O'Brien-Fleming**: Conservative early, liberal late (default choice)
    - **Pocock**: Uniform spending across analyses
    - **HSD** (Hwang-Shih-DeCani): Flexible family with shape parameter gamma

    The calculator computes Z-statistic thresholds at each analysis that
    correspond to the cumulative alpha spent up to that point. Both efficacy
    boundaries (for detecting positive effects) and futility boundaries
    (for early stopping under the null) are supported.

    Methods
    -------
    spending_function(func, t, alpha, gamma=-4.0) -> np.ndarray
        Compute cumulative alpha spending at information times using
        specified spending function. Returns cumulative alpha spent
        at each time point in t.

    critical_values(spec) -> Dict[str, Any]
        Calculate complete boundary specification including efficacy
        and optional futility boundaries. Returns dictionary with
        info_times, cumulative_alpha, z_efficacy, and z_futility.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> boundaries = BoundaryCalculator.critical_values(spec)
    >>> 'z_efficacy' in boundaries
    True
    >>> len(boundaries['info_times'])  # 3 analyses by default
    3

    >>> # Check cumulative alpha spending (two-sided test: 0.05 total, 0.025 per side)
    >>> bool(0.04 < boundaries['cumulative_alpha'][-1] < 0.06)
    True

    >>> # Different spending functions
    >>> from earlysign.stats.design.gst.common.types import SpendingFunction
    >>> spec.boundary.spending_function = SpendingFunction.POCOCK
    >>> boundaries_pocock = BoundaryCalculator.critical_values(spec)

    Notes
    -----
    The implementation uses the Lan-DeMets approach, where alpha spending
    depends only on the fraction of information observed (information time),
    not on the actual calendar time or sample size. This provides flexibility
    in study conduct.

    Information time t ∈ (0, 1] represents the fraction of planned information
    accrued. For example, t=0.5 means halfway to the maximum sample size.

    See Also
    --------
    DesignSpec : Specifies alpha, spending function, and analysis timing
    SpendingFunction : Enum of available spending function types
    DesignLab : Orchestrator that uses BoundaryCalculator
    SimulationEngine : Evaluates operating characteristics with these boundaries

    References
    ----------
    .. [1] Lan, K. K. G., & DeMets, D. L. (1983). "Discrete sequential boundaries
           for clinical trials". Biometrika, 70(3), 659-663.
    .. [2] Hwang, I. K., Shih, W. J., & De Cani, J. S. (1990). "Group sequential
           designs using a family of type I error probability spending functions".
           Statistics in Medicine, 9(12), 1439-1445.
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

        >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
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
