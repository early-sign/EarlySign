"""Effect size and sample size calculations for two-sample t-tests.

This module provides concrete implementation of effect calculations
specific to comparing two population means using a t-test.
"""

from typing import Dict

import numpy as np

from earlysign.stats.design.gst.common.config import DesignSpec, MeansDesignSpec
from earlysign.stats.schemes.base.effects import EffectCalculator
from earlysign.stats.schemes.two_means.util import (
    compute_standard_error as compute_se_means,
)


class MeansEffectCalculator(EffectCalculator):
    """Effect size and sample size calculations for two-sample t-tests.

    This calculator implements formulas for comparing two population means
    using a t-test (or Z-test for large samples) with pooled variance estimate.
    It handles unequal allocation ratios and computes the standardized effect
    (drift) for use in power calculations and simulations.

    The standardized effect is computed as::

        δ = (μ_treatment - μ_control) / SE_pooled

    where SE_pooled assumes a common standard deviation σ::

        SE_pooled = σ * sqrt(1/n_control + 1/n_treatment)

    Methods
    -------
    standardized_effect(spec, info_time) -> float
        Compute standardized mean difference at information time.
    sample_sizes(spec) -> Dict[str, np.ndarray]
        Compute n_control, n_treatment, n_total at each analysis.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import MeansDesignSpec
    >>> spec = MeansDesignSpec()
    >>> spec.effect.mean_control = 100.0
    >>> spec.effect.mean_treatment = 105.0  # 5 unit increase
    >>> spec.effect.std_dev = 15.0
    >>> calc = MeansEffectCalculator()
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
    **Pooled Variance Assumption**

    This calculator assumes equal variances (homoscedasticity) in both groups.
    The pooled standard error uses the common standard deviation σ provided
    in the design specification.

    **Information Accrual**

    Information accrual is proportional to sample size for means tests,
    so information time directly corresponds to the fraction of planned
    sample accumulated:

        information_time = n_accumulated / n_total

    **Effect Size Specification**

    The effect can be specified in two ways:

    1. Absolute difference: mean_control and mean_treatment
    2. Effect size: mean_control and effect_size (Cohen's d-like measure)

    See Also
    --------
    MeansDesignSpec : Specifies means effect parameters
    earlysign.stats.schemes.two_proportions.essentials.effects.ProportionsEffectCalculator
        Similar calculator for binary outcomes
    earlysign.stats.design.gst.simulation.SimulationEngine
        Uses standardized_effect for power simulation
    """

    def standardized_effect(self, spec: DesignSpec, info_time: float) -> float:
        """Calculate standardized effect for means test.

        Computes the drift parameter for the Brownian motion representation
        of the test statistic. This is the expected Z-statistic divided by
        the square root of information time under the alternative hypothesis.

        Parameters
        ----------
        spec : DesignSpec
            Means design specification containing effect parameters
            (mean_control, mean_treatment or effect_size, std_dev) and
            sample size settings.
        info_time : float
            Information time in (0, 1], representing the fraction of total
            planned sample size accumulated.

        Returns
        -------
        float
            Standardized mean difference (delta), computed as:
            (μ_treatment - μ_control) / SE_pooled(n_control, n_treatment)

        Examples
        --------
        >>> from earlysign.stats.design.gst.common.config import MeansDesignSpec
        >>> spec = MeansDesignSpec()
        >>> spec.effect.mean_control = 100.0
        >>> spec.effect.effect_size = 0.5  # Cohen's d = 0.5
        >>> spec.effect.std_dev = 10.0
        >>> calc = MeansEffectCalculator()
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
        The function uses pooled standard error assuming equal variances:

            SE = σ * sqrt(1/n_control + 1/n_treatment)

        This is standard practice in group sequential testing for continuous
        outcomes.
        """
        assert isinstance(spec, MeansDesignSpec)

        mu_A = spec.effect.mean_control
        mu_B = (
            spec.effect.mean_treatment
            if spec.effect.mean_treatment is not None
            else mu_A + (spec.effect.effect_size or 0.0)
        )
        sigma = spec.effect.std_dev

        # Sample sizes at this information time
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses
        n_A = int(n_total_A * info_time)
        n_B = int(n_A * spec.allocation.alloc_ratio)

        if n_A == 0 or n_B == 0:
            return 0.0

        # Use common standard error calculation
        se = compute_se_means(n_A, n_B, sigma, pooled=True)

        if se == 0:
            return 0.0

        return float((mu_B - mu_A) / se)

    def sample_sizes(self, spec: DesignSpec) -> Dict[str, np.ndarray]:
        """Calculate sample sizes at each analysis.

        Determines the number of observations in control and treatment groups
        at each planned interim analysis and final analysis, based on the
        information times and allocation ratio.

        Parameters
        ----------
        spec : DesignSpec
            Means design specification containing sample size requirements
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
        >>> from earlysign.stats.design.gst.common.config import MeansDesignSpec
        >>> spec = MeansDesignSpec()
        >>> spec.sequential.n_analyses = 3
        >>> spec.allocation.alloc_ratio = 1.0  # Equal allocation
        >>> spec.sample_size.n_per_analysis = 100
        >>> calc = MeansEffectCalculator()
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
        assert isinstance(spec, MeansDesignSpec)

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
