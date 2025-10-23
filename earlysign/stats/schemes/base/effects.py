"""Abstract base class for effect size and sample size calculations.

This module defines the protocol that all test-specific effect calculators
must implement. Different statistical tests (two proportions, two means,
time-to-event) require different formulations for standardized effect sizes
and sample size calculations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from earlysign.stats.design.gst.common.config import DesignSpec


class EffectCalculator(ABC):
    """Abstract base class for test-specific effect size and sample size calculations.

    Different statistical tests (two proportions, two means, time-to-event)
    require different formulations for:

    1. **Standardized effect size (drift parameter)**: The drift parameter for
       Brownian motion representation of the test statistic. This determines
       the mean of the Z-statistic under the alternative hypothesis.

    2. **Sample size at each analysis**: May vary by allocation ratio and
       information time. For some tests (e.g., survival), information is
       based on events rather than sample size.

    3. **Information accrual**: The relationship between sample size/events
       and statistical information depends on the test type.

    This protocol defines the interface that all test-specific calculators
    must implement. Concrete implementations handle the mathematical details
    for each test type.

    Methods
    -------
    standardized_effect(spec, info_time) -> float
        Compute the standardized effect (drift) at a given information time.
        This determines the mean of the Z-statistic under the alternative
        hypothesis.

    sample_sizes(spec) -> Dict[str, np.ndarray]
        Compute sample sizes at each planned analysis. Returns dictionary
        with keys like 'n_control', 'n_treatment', 'n_total', containing
        arrays of length n_analyses.

    Examples
    --------
    >>> from earlysign.stats.design.gst.schemes.two_proportions.config import ProportionsDesignSpec
    >>> from earlysign.stats.schemes.two_proportions.essentials.effects import (
    ...     ProportionsEffectCalculator
    ... )
    >>> spec = ProportionsDesignSpec()
    >>> calc = ProportionsEffectCalculator()
    >>> effect = calc.standardized_effect(spec, info_time=1.0)
    >>> effect > 0  # Positive effect
    True
    >>> sizes = calc.sample_sizes(spec)
    >>> 'n_total' in sizes
    True

    See Also
    --------
    earlysign.stats.schemes.two_proportions.essentials.effects.ProportionsEffectCalculator
        For two-proportion tests
    earlysign.stats.schemes.two_means.essentials.effects.MeansEffectCalculator
        For two-sample t-tests
    earlysign.stats.design.gst.simulation.SimulationEngine
        Uses standardized_effect for drift
    earlysign.stats.design.gst.core.DesignLab
        Orchestrates effect calculations

    Notes
    -----
    The standardized effect is typically computed as::

        δ = (effect_size) / SE(effect_size)

    where the standard error depends on the sample size and the test type.
    For group sequential testing, this standardized effect corresponds to
    the drift parameter μ in the Brownian motion representation::

        Z(t) = μ√t + B(t)

    where Z(t) is the standardized test statistic at information time t,
    and B(t) is standard Brownian motion.
    """

    @abstractmethod
    def standardized_effect(self, spec: "DesignSpec", info_time: float) -> float:
        """Calculate standardized effect at given information time.

        The standardized effect represents the drift parameter (μ) in the
        Brownian motion model of the test statistic. It should be computed
        as the effect size divided by its standard error at the given
        information time.

        Parameters
        ----------
        spec : DesignSpec
            Design specification containing effect parameters, sample sizes,
            and allocation ratios.
        info_time : float
            Information time in (0, 1], representing the fraction of total
            planned information. For sample-based tests, this corresponds to
            the fraction of planned sample size. For event-based tests (e.g.,
            survival), this is the fraction of planned events.

        Returns
        -------
        float
            Standardized effect size (drift parameter μ). This is the expected
            value of the Z-statistic scaled by √information_time under the
            alternative hypothesis.

        Examples
        --------
        >>> from earlysign.stats.design.gst.schemes.two_proportions.config import ProportionsDesignSpec
        >>> from earlysign.stats.schemes.two_proportions.essentials.effects import (
        ...     ProportionsEffectCalculator
        ... )
        >>> spec = ProportionsDesignSpec()
        >>> spec.effect.p_control = 0.10
        >>> spec.effect.delta = 0.05
        >>> calc = ProportionsEffectCalculator()
        >>> drift = calc.standardized_effect(spec, info_time=0.5)
        >>> drift > 0
        True
        """
        pass

    @abstractmethod
    def sample_sizes(self, spec: "DesignSpec") -> Dict[str, np.ndarray]:
        """Calculate sample sizes at each analysis.

        Compute the number of observations or events at each planned analysis
        based on the design specification. The exact meaning depends on the
        test type:

        - **Sample-based tests** (proportions, means): Actual sample sizes
        - **Event-based tests** (survival): Number of events and total enrolled

        Parameters
        ----------
        spec : DesignSpec
            Design specification containing sample size requirements and
            information time settings.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with sample size information. Common keys include:

            - 'n_control': Sample sizes for control group at each analysis
            - 'n_treatment': Sample sizes for treatment group at each analysis
            - 'n_total': Total sample sizes at each analysis
            - 'info_fraction': Information fractions at each analysis
            - 'events': Number of events (for survival tests)

            All arrays have length equal to the number of analyses.

        Examples
        --------
        >>> from earlysign.stats.design.gst.schemes.two_proportions.config import ProportionsDesignSpec
        >>> from earlysign.stats.schemes.two_proportions.essentials.effects import (
        ...     ProportionsEffectCalculator
        ... )
        >>> spec = ProportionsDesignSpec()
        >>> spec.sequential.n_analyses = 3
        >>> calc = ProportionsEffectCalculator()
        >>> sizes = calc.sample_sizes(spec)
        >>> len(sizes['n_total'])
        3
        >>> 'n_control' in sizes and 'n_treatment' in sizes
        True
        """
        pass
