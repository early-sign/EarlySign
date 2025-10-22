"""Boundary value calculation for group sequential designs."""

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from earlysign.stats.common.group_sequential.essentials.spending import (
    hsd_spending,
    obf_spending,
    pocock_spending,
)
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
    >>> bool(0.024 < boundaries['cumulative_alpha'][-1] < 0.026)
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

        # Use common alpha spending functions
        if spec.boundary.spending_function == SpendingFunction.OBRIEN_FLEMING:
            cumulative_alpha = np.array(
                [
                    obf_spending(
                        ti, alpha_total, tails=1 if spec.test.sided == "one" else 2
                    )
                    for ti in t
                ]
            )
        elif spec.boundary.spending_function == SpendingFunction.POCOCK:
            cumulative_alpha = np.array([pocock_spending(ti, alpha_total) for ti in t])
        elif spec.boundary.spending_function == SpendingFunction.HSD:
            cumulative_alpha = np.array(
                [hsd_spending(ti, alpha_total, spec.boundary.hsd_gamma) for ti in t]
            )
        else:
            raise ValueError(
                f"Unknown spending function: {spec.boundary.spending_function}"
            )

        # Incremental alpha at each analysis
        alpha_increments = np.diff(np.concatenate([[0.0], cumulative_alpha]))

        # Critical Z-values
        # Note: This uses a simple approximation. For exact boundaries accounting
        # for correlation between analyses, numerical integration would be needed.
        # The simulation engine will account for the correlation structure properly.
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

                # Use common alpha spending functions for futility
                if spec.boundary.futility_spending == SpendingFunction.OBRIEN_FLEMING:
                    cumulative_beta = np.array(
                        [
                            obf_spending(
                                ti, beta, tails=1 if spec.test.sided == "one" else 2
                            )
                            for ti in t
                        ]
                    )
                elif spec.boundary.futility_spending == SpendingFunction.POCOCK:
                    cumulative_beta = np.array([pocock_spending(ti, beta) for ti in t])
                elif spec.boundary.futility_spending == SpendingFunction.HSD:
                    cumulative_beta = np.array(
                        [hsd_spending(ti, beta, spec.boundary.hsd_gamma) for ti in t]
                    )
                else:
                    raise ValueError(
                        f"Unknown futility spending function: {spec.boundary.futility_spending}"
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
