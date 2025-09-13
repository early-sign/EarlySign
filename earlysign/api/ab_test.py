"""
earlysign.api.ab_test
====================

Comprehensive A/B testing facade with business-oriented interfaces.

This module provides all A/B testing functionality using familiar terminology
from the experimentation domain, including interim analysis and guardrail monitoring.

Examples
--------
>>> from earlysign.api.ab_test import interim_analysis, guardrail_monitoring
>>> from earlysign.runtime.runners import SequentialRunner
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>>
>>> # A/B Test with interim analysis
>>> experiment = interim_analysis("checkout_test", alpha=0.05, looks=4)
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "checkout_test")
>>> runner = SequentialRunner(experiment, ledger)
>>>
>>> # Guardrail monitoring
>>> guardrail = guardrail_monitoring("payment_safety", sensitivity="balanced")
"""

from __future__ import annotations
from typing import Optional, Literal, Dict, Any, List
from dataclasses import dataclass, field

from earlysign.stats.schemes.two_proportions.experiments import TwoPropGSTTemplate
from earlysign.stats.schemes.two_proportions.experiments import TwoPropSafeTemplate


def interim_analysis(
    experiment_id: str,
    alpha: float = 0.05,
    looks: int = 4,
    spending: Literal["conservative", "aggressive"] = "conservative",
) -> TwoPropGSTTemplate:
    """
    Create an A/B test experiment with interim analysis capabilities.

    This function sets up a two-proportions experiment that allows for
    early stopping based on interim analysis at predetermined looks.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    alpha : float, default=0.05
        Overall Type I error rate (significance level)
    looks : int, default=4
        Number of interim analyses planned
    spending : {"conservative", "aggressive"}, default="conservative"
        Spending function approach:
        - "conservative": O'Brien-Fleming (saves alpha for later looks)
        - "aggressive": Pocock (equal spending across looks)

    Returns
    -------
    TwoPropGSTTemplate
        A configured experiment module ready for execution

    Examples
    --------
    >>> # Conservative approach (O'Brien-Fleming)
    >>> experiment = interim_analysis("test_v1", alpha=0.05, looks=3, spending="conservative")
    >>>
    >>> # Aggressive approach (Pocock)
    >>> experiment = interim_analysis("test_v2", alpha=0.05, looks=3, spending="aggressive")
    """
    # Map business terms to technical terms
    spending_function_map = {
        "conservative": "obf",  # O'Brien-Fleming
        "aggressive": "pocock",  # Pocock
    }

    return TwoPropGSTTemplate(
        experiment_id=experiment_id,
        alpha_total=alpha,
        looks=looks,
        spending_function=spending_function_map[spending],
    )


def fixed_sample_test(
    experiment_id: str,
    alpha: float = 0.05,
) -> TwoPropGSTTemplate:
    """
    Create a traditional fixed-sample A/B test.

    This is equivalent to an interim analysis with only one look
    at the end of data collection.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    alpha : float, default=0.05
        Type I error rate (significance level)

    Returns
    -------
    TwoPropGSTTemplate
        A configured experiment module for fixed-sample testing

    Examples
    --------
    >>> experiment = fixed_sample_test("checkout_final", alpha=0.05)
    """
    return TwoPropGSTTemplate(
        experiment_id=experiment_id,
        alpha_total=alpha,
        looks=1,
        spending_function="obf",  # Doesn't matter for single look
    )


def guardrail_monitoring(
    experiment_id: str,
    alpha: float = 0.05,
    sensitivity: Literal["conservative", "balanced", "sensitive"] = "balanced",
    prior_strength: Optional[float] = None,
) -> TwoPropSafeTemplate:
    """
    Create a safe test for guardrail monitoring.

    Safe tests allow for continuous monitoring without inflating
    Type I error, making them ideal for guardrail scenarios where
    you need to detect problems quickly while maintaining statistical validity.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the guardrail test
    alpha : float, default=0.05
        Type I error rate for the overall monitoring period
    sensitivity : {"conservative", "balanced", "sensitive"}, default="balanced"
        Detection sensitivity:
        - "conservative": Lower false positives, may miss subtle effects
        - "balanced": Good balance between sensitivity and specificity
        - "sensitive": Higher sensitivity, may have more false positives
    prior_strength : float, optional
        Strength of prior beliefs. If None, uses defaults based on sensitivity

    Returns
    -------
    TwoPropSafeTemplate
        A configured safe testing module ready for continuous monitoring

    Examples
    --------
    >>> # Balanced guardrail for general monitoring
    >>> guardrail = guardrail_monitoring("conversion_monitor", sensitivity="balanced")
    >>>
    >>> # Sensitive guardrail for critical metrics
    >>> critical = guardrail_monitoring("payment_failures", sensitivity="sensitive", alpha=0.01)
    """
    # Map business sensitivity levels to prior parameters
    sensitivity_priors = {
        "conservative": {"alpha": 0.5, "beta": 0.5},  # Weak prior, conservative
        "balanced": {"alpha": 1.0, "beta": 1.0},  # Uniform prior, balanced
        "sensitive": {"alpha": 2.0, "beta": 2.0},  # Stronger prior, more sensitive
    }

    if prior_strength is not None:
        # Custom prior strength
        prior_params = {"alpha": prior_strength, "beta": prior_strength}
    else:
        prior_params = sensitivity_priors[sensitivity]

    return TwoPropSafeTemplate(
        experiment_id=experiment_id,
        alpha_total=alpha,
        prior_params=prior_params,
    )


def continuous_monitoring(
    experiment_id: str,
    alpha: float = 0.05,
    baseline_assumption: Literal["no_effect", "small_effect", "custom"] = "no_effect",
    custom_prior: Optional[Dict[str, float]] = None,
) -> TwoPropSafeTemplate:
    """
    Set up continuous monitoring for ongoing experiments or features.

    This is optimized for long-running monitoring scenarios where
    you want to detect changes from a baseline over time.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the monitoring setup
    alpha : float, default=0.05
        Type I error rate for detecting changes
    baseline_assumption : {"no_effect", "small_effect", "custom"}, default="no_effect"
        Assumption about baseline behavior:
        - "no_effect": Assume no difference initially
        - "small_effect": Assume small differences are normal
        - "custom": Use custom prior parameters
    custom_prior : dict, optional
        Custom prior parameters when baseline_assumption="custom"
        Should have keys "alpha" and "beta"

    Returns
    -------
    TwoPropSafeTemplate
        A configured module for continuous monitoring

    Examples
    --------
    >>> # Monitor assuming no baseline difference
    >>> monitor = continuous_monitoring("feature_impact", baseline_assumption="no_effect")
    >>>
    >>> # Monitor with custom baseline assumptions
    >>> custom_monitor = continuous_monitoring(
    ...     "custom_metric",
    ...     baseline_assumption="custom",
    ...     custom_prior={"alpha": 3.0, "beta": 1.0}
    ... )
    """
    baseline_priors = {
        "no_effect": {"alpha": 1.0, "beta": 1.0},  # Uniform, no effect assumed
        "small_effect": {
            "alpha": 0.5,
            "beta": 0.5,
        },  # Weak prior, allows for small effects
    }

    if baseline_assumption == "custom":
        if custom_prior is None:
            raise ValueError(
                "custom_prior must be provided when baseline_assumption='custom'"
            )
        prior_params = custom_prior
    else:
        prior_params = baseline_priors[baseline_assumption]

    return TwoPropSafeTemplate(
        experiment_id=experiment_id,
        alpha_total=alpha,
        prior_params=prior_params,
    )


# =============================================================================
# Advanced A/B Testing with Guardrails
# =============================================================================


@dataclass
class GuardrailConfig:
    """
    Configuration for guardrail metrics in A/B testing.

    Guardrails are safety metrics that monitor for negative side effects
    during an experiment, such as increased load times or reduced engagement.

    Parameters
    ----------
    name : str
        Name of the guardrail metric (e.g., "loadtime", "bounce_rate")
    alpha : float
        Significance level allocated to this guardrail
    method : str, default="safe_test"
        Statistical method: "safe_test" for continuous monitoring, "gst" for interim analysis
    alpha_prior : float, default=1.0
        Prior parameter for safe testing (Beta distribution alpha)
    beta_prior : float, default=1.0
        Prior parameter for safe testing (Beta distribution beta)

    Examples
    --------
    >>> # Standard guardrail for load time monitoring
    >>> loadtime_guard = GuardrailConfig(
    ...     name="loadtime",
    ...     alpha=0.025,
    ...     method="safe_test"
    ... )

    >>> # Engagement guardrail with custom prior
    >>> engagement_guard = GuardrailConfig(
    ...     name="engagement",
    ...     alpha=0.025,
    ...     alpha_prior=2.0,
    ...     beta_prior=1.0
    ... )
    """

    name: str
    alpha: float
    method: str = "safe_test"
    alpha_prior: float = 1.0
    beta_prior: float = 1.0

    def validate(self) -> None:
        """Validate guardrail configuration."""
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError(f"Alpha must be in (0,1), got {self.alpha}")
        if self.method not in ["safe_test", "gst"]:
            raise ValueError(f"Method must be 'safe_test' or 'gst', got {self.method}")
        if self.alpha_prior <= 0 or self.beta_prior <= 0:
            raise ValueError("Prior parameters must be positive")


@dataclass
class ABTestResult:
    """
    Results from a comprehensive A/B test with primary endpoint and guardrails.

    Attributes
    ----------
    experiment_id : str
        Identifier for the experiment
    primary_decision : str
        Decision on primary endpoint: "stop_efficacy", "stop_futility", "continue"
    primary_statistic : float
        Test statistic value for primary endpoint
    primary_threshold : float
        Decision threshold for primary endpoint
    guardrail_results : Dict[str, Dict[str, Any]]
        Results for each guardrail metric
    should_stop : bool
        Overall recommendation to stop the experiment
    stop_reason : str
        Reason for stopping recommendation
    look_number : int
        Current analysis look number
    """

    experiment_id: str
    primary_decision: str
    primary_statistic: float
    primary_threshold: float
    guardrail_results: Dict[str, Dict[str, Any]]
    should_stop: bool
    stop_reason: str
    look_number: int


def ab_test_with_guardrails(
    experiment_id: str,
    primary_alpha: float = 0.05,
    guardrails: Optional[List[GuardrailConfig]] = None,
    looks: int = 5,
    spending: Literal["conservative", "aggressive"] = "conservative",
    adaptive_info: bool = True,
    target_n_per_arm: int = 1000,
) -> "ABTestExperiment":
    """
    Create a comprehensive A/B test with primary endpoint and guardrail monitoring.

    This creates a sophisticated A/B testing setup that monitors a primary business
    metric (e.g., conversion rate) while simultaneously checking guardrail metrics
    (e.g., load times, engagement) to ensure the experiment doesn't cause harm.

    The primary endpoint uses the full alpha for interim analysis with group sequential
    testing, while guardrails use continuous safe testing for real-time monitoring.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the A/B test
    primary_alpha : float, default=0.05
        Significance level for the primary business metric
    guardrails : List[GuardrailConfig], optional
        List of guardrail metrics to monitor for safety
    looks : int, default=5
        Number of planned interim analyses for the primary endpoint
    spending : str, default="conservative"
        Alpha spending strategy: "conservative" or "aggressive"
    adaptive_info : bool, default=True
        Whether to adapt information timing based on actual sample sizes
    target_n_per_arm : int, default=1000
        Target sample size per treatment arm

    Returns
    -------
    ABTestExperiment
        Configured experiment ready for execution

    Examples
    --------
    Simple A/B test with load time guardrail:

    >>> guardrails = [GuardrailConfig("loadtime", alpha=0.025)]
    >>> # experiment = ab_test_with_guardrails(
    >>> #     experiment_id="checkout_optimization",
    >>> #     primary_alpha=0.05,
    >>> #     guardrails=guardrails,
    >>> #     looks=4
    >>> # )

    Complex test with multiple guardrails:

    >>> guardrails = [
    ...     GuardrailConfig("loadtime", alpha=0.02),
    ...     GuardrailConfig("bounce_rate", alpha=0.02),
    ...     GuardrailConfig("engagement", alpha=0.01)
    ... ]
    >>> # experiment = ab_test_with_guardrails(
    >>> #     experiment_id="homepage_redesign",
    >>> #     guardrails=guardrails,
    >>> #     adaptive_info=True
    >>> # )
    """
    if guardrails is None:
        guardrails = []

    return ABTestExperiment(
        experiment_id=experiment_id,
        primary_alpha=primary_alpha,
        guardrails=guardrails,
        looks=looks,
        spending=spending,
        adaptive_info=adaptive_info,
        target_n_per_arm=target_n_per_arm,
    )


# Import the implementation from the moved multi_metric content
from earlysign.runtime.experiment_template import ExperimentTemplate
from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace, ExperimentId
from earlysign.stats.common.group_sequential import AdaptiveInfoTime
from earlysign.stats.schemes.two_proportions.statistics import (
    WaldZStatistic,
    LanDeMetsBoundary,
    PeekSignaler,
    BetaBinomialEValue,
    SafeThreshold,
    SafeSignaler,
)
from earlysign.stats.schemes.two_proportions.core import TwoPropObservation


@dataclass
class ABTestExperiment(ExperimentTemplate):
    """
    Comprehensive A/B testing experiment with primary endpoint and guardrails.

    This class implements a sophisticated A/B testing framework that coordinates:
    - Primary business metric analysis using group sequential testing
    - Multiple guardrail metrics using safe testing for continuous monitoring
    - Adaptive information timing based on real-world sample sizes
    - Proper alpha allocation between primary and guardrail metrics

    The design follows enterprise A/B testing best practices where the primary
    metric gets full statistical power while guardrails provide safety monitoring
    without compromising the main analysis.

    Attributes
    ----------
    experiment_id : str
        Unique identifier for the experiment
    primary_alpha : float, default=0.05
        Significance level for primary endpoint
    guardrails : List[GuardrailConfig], default=[]
        Guardrail metric configurations
    looks : int, default=5
        Number of planned interim analyses
    spending : str, default="conservative"
        Alpha spending function strategy
    target_n_per_arm : int, default=1000
        Target sample size per treatment arm
    adaptive_info : bool, default=True
        Enable adaptive information timing

    Examples
    --------
    Configure experiment with CTR primary and safety guardrails:

    >>> guardrails = [
    ...     GuardrailConfig("loadtime", alpha=0.025),
    ...     GuardrailConfig("bounce_rate", alpha=0.025)
    ... ]
    >>> # experiment = ABTestExperiment(
    >>> #     experiment_id="feature_launch",
    >>> #     primary_alpha=0.05,
    >>> #     guardrails=guardrails
    >>> # )
    """

    experiment_id: ExperimentId
    primary_alpha: float = 0.05
    guardrails: List[GuardrailConfig] = field(default_factory=list)
    looks: int = 5
    spending: str = "conservative"
    target_n_per_arm: int = 1000
    adaptive_info: bool = True

    # Internal components (populated in __post_init__)
    primary_stat: Any = field(init=False)
    primary_signaler: Any = field(init=False)
    guardrail_components: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    ingestor: Any = field(init=False)
    adaptive_info_time: Optional[AdaptiveInfoTime] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize all statistical components."""
        # Validate configuration
        self._validate_config()

        # Primary endpoint analysis (two-sided GST)
        self.primary_stat = WaldZStatistic(tag_stats="stat:primary_ctr")
        self.primary_signaler = PeekSignaler()

        # Initialize guardrail components
        for guardrail in self.guardrails:
            self.guardrail_components[guardrail.name] = (
                self._create_guardrail_components(guardrail)
            )

        # Data ingestion for all metrics
        self.ingestor = TwoPropObservation(tag_obs="obs:ab_test")

        # Information time adaptation
        if self.adaptive_info:
            self.adaptive_info_time = AdaptiveInfoTime(
                initial_target=self.target_n_per_arm, looks=self.looks
            )

    def _validate_config(self) -> None:
        """Validate experiment configuration."""
        # Validate primary alpha
        if not 0 < self.primary_alpha < 1:
            raise ValueError(
                f"Primary alpha must be in (0,1), got {self.primary_alpha}"
            )

        # Validate guardrails
        total_guardrail_alpha = sum(g.alpha for g in self.guardrails)
        if total_guardrail_alpha > 0.5:  # Sanity check
            raise ValueError(
                f"Total guardrail alpha seems high: {total_guardrail_alpha}"
            )

        for guardrail in self.guardrails:
            guardrail.validate()

    def _create_guardrail_components(self, config: GuardrailConfig) -> Dict[str, Any]:
        """Create statistical components for a guardrail metric."""
        components: Dict[str, Any] = {}

        if config.method == "safe_test":
            components["statistic"] = BetaBinomialEValue(
                tag_stats=f"stat:{config.name}_safe",
                alpha_prior=config.alpha_prior,
                beta_prior=config.beta_prior,
            )
            components["criteria"] = SafeThreshold(alpha_level=config.alpha)
            components["signaler"] = SafeSignaler()
        elif config.method == "gst":
            # GST guardrails (less common but possible)
            components["statistic"] = WaldZStatistic(
                tag_stats=f"stat:{config.name}_gst"
            )
            components["signaler"] = PeekSignaler()
        else:
            raise ValueError(f"Unknown method: {config.method}")

        return components

    def configure_components(self) -> Dict[str, Any]:
        """Configure all statistical components for the experiment."""
        components_dict = {
            "primary_statistic": self.primary_stat,
            "primary_signaler": self.primary_signaler,
            "ingestor": self.ingestor,
        }

        if self.adaptive_info:
            components_dict["adaptive_info_time"] = self.adaptive_info_time

        for guardrail in self.guardrails:
            guardrail_components = self.guardrail_components.get(guardrail.name, {})
            components_dict[f"{guardrail.name}_statistic"] = guardrail_components.get(
                "statistic"
            )
            components_dict[f"{guardrail.name}_criteria"] = guardrail_components.get(
                "criteria"
            )
            components_dict[f"{guardrail.name}_signaler"] = guardrail_components.get(
                "signaler"
            )

        return components_dict
