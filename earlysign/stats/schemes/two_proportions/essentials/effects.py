"""Effect size and sample size calculations for two-proportion tests.

This module provides concrete implementation of effect calculations
specific to comparing two binomial proportions using a Z-test.
"""

from typing import Dict

import numpy as np

from earlysign.stats.design.gst.common.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.schemes.base.effects import EffectCalculator
from earlysign.stats.schemes.two_proportions.util import (
    compute_standard_error as compute_se_proportions,
)


class ProportionsEffectCalculator(EffectCalculator):
    """Effect size and sample size calculations for two-proportion tests.

    This calculator implements the standard formulas for comparing two
    binomial proportions using a Z-test with pooled variance estimate.
    It handles unequal allocation ratios and computes the standardized
    effect (drift) for use in power calculations and simulations.

    The standardized effect is computed as::

        δ = (p_treatment - p_control) / SE_pooled

    where SE_pooled uses the pooled proportion under H₀::

        p_pooled = (p_control + p_treatment) / 2
        SE_pooled = sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    Methods
    -------
    standardized_effect(spec, info_time) -> float
        Compute standardized difference at information time.
    sample_sizes(spec) -> Dict[str, np.ndarray]
        Compute n_control, n_treatment, n_total at each analysis.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> spec.effect.p_control = 0.10
    >>> spec.effect.delta = 0.05  # 5% absolute increase
    >>> calc = ProportionsEffectCalculator()
    >>> effect = calc.standardized_effect(spec, 1.0)
    >>> effect > 0  # Positive effect
    True
    >>> sizes = calc.sample_sizes(spec)
    >>> 'n_total' in sizes
    True
    >>> len(sizes['n_total'])  # One value per analysis
    3

    Notes
    -----
    **Pooled vs. Unpooled Variance**

    The pooled variance estimator is used for consistency with the null
    distribution in group sequential testing. This differs from the
    unpooled estimator sometimes used in fixed-sample tests.

    **Information Accrual**

    Information accrual is proportional to sample size for proportion tests,
    so information time directly corresponds to the fraction of planned
    sample accumulated:

        information_time = n_accumulated / n_total

    See Also
    --------
    ProportionsDesignSpec : Specifies proportions effect parameters
    earlysign.stats.schemes.two_means.essentials.effects.MeansEffectCalculator
        Similar calculator for continuous outcomes
    earlysign.stats.design.gst.simulation.SimulationEngine
        Uses standardized_effect for power simulation
    """

    def standardized_effect(self, spec: DesignSpec, info_time: float) -> float:
        """Calculate standardized effect (delta) for proportions test.

        Computes the drift parameter for the Brownian motion representation
        of the test statistic. This is the expected Z-statistic divided by
        the square root of information time under the alternative hypothesis.

        Parameters
        ----------
        spec : DesignSpec
            Proportions design specification containing effect parameters
            (p_control, delta) and sample size settings.
        info_time : float
            Information time in (0, 1], representing the fraction of total
            planned sample size accumulated.

        Returns
        -------
        float
            Standardized difference (delta), computed as:
            (p_treatment - p_control) / SE_pooled(n_control, n_treatment)

        Examples
        --------
        >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
        >>> spec = ProportionsDesignSpec()
        >>> spec.effect.p_control = 0.10
        >>> spec.effect.delta = 0.05
        >>> calc = ProportionsEffectCalculator()
        >>> # At 50% information time
        >>> drift = calc.standardized_effect(spec, 0.5)
        >>> drift > 0
        True
        >>> # At 100% information time (full sample)
        >>> drift_full = calc.standardized_effect(spec, 1.0)
        >>> drift_full > drift
        True

        Notes
        -----
        The function uses the pooled proportion for computing the standard error:

            p_pooled = (p_control + p_treatment) / 2

        This ensures consistency with the null hypothesis (H₀: p_control = p_treatment)
        and is standard practice in group sequential testing.
        """
        assert isinstance(spec, ProportionsDesignSpec)

        p_A = spec.effect.p_control
        p_B = spec.effect.get_treatment_proportion()

        # Sample sizes at this information time
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses
        n_A = int(n_total_A * info_time)
        n_B = int(n_A * spec.allocation.alloc_ratio)

        if n_A == 0 or n_B == 0:
            return 0.0

        # Use pooled proportion for H0
        p_pooled = (p_A + p_B) / 2.0

        # Use common standard error calculation
        se = compute_se_proportions(n_A, n_B, p_pooled, p_pooled, pooled=True)

        if se == 0:
            return 0.0

        # Standardized difference
        return float((p_B - p_A) / se)

    def sample_sizes(self, spec: DesignSpec) -> Dict[str, np.ndarray]:
        """Calculate sample sizes at each analysis.

        Determines the number of observations in control and treatment groups
        at each planned interim analysis and final analysis, based on the
        information times and allocation ratio.

        Parameters
        ----------
        spec : DesignSpec
            Proportions design specification containing sample size requirements
            and information time settings.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:

            - 'n_control': Control group sample sizes at each analysis
            - 'n_treatment': Treatment group sample sizes at each analysis
            - 'n_total': Total sample sizes at each analysis
            - 'info_fraction': Information fractions at each analysis

            All arrays have length equal to spec.sequential.n_analyses.

        Examples
        --------
        >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
        >>> spec = ProportionsDesignSpec()
        >>> spec.sequential.n_analyses = 3
        >>> spec.allocation.alloc_ratio = 1.0  # Equal allocation
        >>> spec.sample_size.n_per_analysis = 100
        >>> calc = ProportionsEffectCalculator()
        >>> sizes = calc.sample_sizes(spec)
        >>> sizes['n_control']  # doctest: +SKIP
        array([ 33, 66, 100])
        >>> sizes['n_treatment']  # doctest: +SKIP
        array([ 33, 66, 100])
        >>> sizes['info_fraction']  # doctest: +SKIP
        array([0.33, 0.67, 1.0])

        Notes
        -----
        **Allocation Ratio**

        The allocation ratio r determines the relative sample sizes:

            n_treatment = r * n_control

        For r = 1, groups have equal sizes. For r = 2, treatment group
        is twice as large as control.

        **Information Times**

        If custom information times are specified, sample sizes are computed
        accordingly. Otherwise, equally-spaced information times are used.
        """
        assert isinstance(spec, ProportionsDesignSpec)

        t = spec.resolved_info_times()
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses

        n_A = (n_total_A * t).astype(int)
        n_B = (n_A * spec.allocation.alloc_ratio).astype(int)

        return {
            "n_control": n_A,
            "n_treatment": n_B,
            "n_total": n_A + n_B,
            "info_fraction": t,
        }
