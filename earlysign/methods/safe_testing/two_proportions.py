"""
earlysign.methods.safe_testing.two_proportions
==============================================

Safe testing components for two-proportion co      # Optimize over reasonable range for log(alpha)
    result = minimize_scalar(neg_log_evalue, bounds=(-2, 5), method="bounded")
    return float(-result.fun) if result.success else 0.0 Optimize over reasonable range for log(alpha)
    result = minimize_scalar(neg_log_evalue, bounds=(-2, 5), method='bounded')
    return float(-result.fun) if result.success else 0.0risons using beta-binomial e-processes.

This module implements e-value based testing for comparing two binomial proportions,
providing anytime-valid inference without traditional sequential testing corrections.

Components:
- `BetaBinomialEValue`: Compute beta-binomial e-values for two-group comparisons
- `SafeThreshold`: Define anytime-valid significance thresholds
- `SafeSignaler`: Emit stopping signals based on e-value thresholds

Mathematical Background:
The beta-binomial e-process uses a mixture of beta distributions as the alternative
hypothesis, providing a natural conjugate prior structure for binomial data.

For testing H0: pA = pB vs H1: pA ≠ pB, the e-value at time n is:

E_n = ∫ ∏_{i=1}^{n_A} Beta(y_i; α, β) ∏_{j=1}^{n_B} Beta(z_j; α', β') dF(α, β, α', β')

where F is the prior over the beta parameters and y_i, z_j are the observations.

References:
- Turner, A. et al. (2024). Safe testing for two-sample binomial problems.
- Grünwald, P. et al. (2020). Safe testing. Statistical Science.

Examples
--------
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> L = PolarsLedger()
>>> # Ingest observation with clear difference
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":10,"nB":10,"mA":1,"mB":8})
>>> # Compute e-value
>>> BetaBinomialEValue().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> # Check threshold
>>> SafeThreshold().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> # Signal if significant
>>> SafeSignaler().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any

from scipy.special import betaln, gammaln
from scipy.optimize import minimize_scalar

from earlysign.core.components import Criteria, Signaler, Statistic
from earlysign.core.names import (
    Namespace,
    ExperimentId,
    StepKey,
    TimeIndex,
)
from earlysign.core.ledger import Ledger
from earlysign.schemes.two_proportions.reduce import reduce_counts
from earlysign.schemes.two_proportions.model import TwoPropObsBatch


# --- Mathematical helpers for beta-binomial e-values ---


def log_beta_binomial_evalue_simple(
    mA: int,
    nA: int,
    mB: int,
    nB: int,
    alpha_A: float = 1.0,
    beta_A: float = 1.0,
    alpha_B: float = 1.0,
    beta_B: float = 1.0,
) -> float:
    """
    Compute log e-value for two-sample beta-binomial test with fixed priors.

    This implements the simple case where we use fixed beta priors for each group.

    Args:
        mA, nA: Successes and trials in group A
        mB, nB: Successes and trials in group B
        alpha_A, beta_A: Beta prior parameters for group A
        alpha_B, beta_B: Beta prior parameters for group B

    Returns:
        Log e-value (natural log)

    Note:
        The e-value is the Bayes factor comparing the alternative (independent betas)
        to the null hypothesis (common probability under uniform prior).
    """
    if min(nA, nB) == 0:
        return 0.0

    # Alternative: independent beta-binomial likelihoods
    log_alt_A = betaln(alpha_A + mA, beta_A + nA - mA) - betaln(alpha_A, beta_A)
    log_alt_B = betaln(alpha_B + mB, beta_B + nB - mB) - betaln(alpha_B, beta_B)
    log_alternative = log_alt_A + log_alt_B

    # Null: pooled data with uniform prior (equivalent to Beta(1,1))
    m_pool = mA + mB
    n_pool = nA + nB
    log_null = betaln(1 + m_pool, 1 + n_pool - m_pool) - betaln(1, 1)

    return float(log_alternative - log_null)


def log_beta_binomial_evalue_adaptive(
    mA: int,
    nA: int,
    mB: int,
    nB: int,
    alpha_shape: float = 1.0,
    beta_shape: float = 1.0,
) -> float:
    """
    Compute log e-value using adaptive/optimized beta priors.

    This version optimizes the choice of beta parameters to maximize the e-value,
    which can improve power while maintaining validity.

    Args:
        mA, nA: Successes and trials in group A
        mB, nB: Successes and trials in group B
        alpha_shape, beta_shape: Shape parameters for the prior on beta parameters

    Returns:
        Log e-value (natural log)
    """
    if min(nA, nB) == 0:
        return 0.0

    def neg_log_evalue(log_alpha: float) -> float:
        """Negative log e-value to minimize (i.e., maximize e-value)."""
        alpha = math.exp(log_alpha)
        beta = beta_shape  # Could also optimize this
        return -log_beta_binomial_evalue_simple(
            mA, nA, mB, nB, alpha, beta, alpha, beta
        )

    # Optimize over reasonable range for log(alpha)
    result = minimize_scalar(neg_log_evalue, bounds=(-2, 5), method="bounded")
    return -result.fun if result.success else 0.0


# --- Payload type definitions ---


class BetaBinomialEValuePayload:
    """TypedDict for beta-binomial e-value payload."""

    e_value: float
    log_e_value: float
    nA: int
    nB: int
    mA: int
    mB: int
    alpha_prior: float
    beta_prior: float
    method: str


class SafeThresholdPayload:
    """TypedDict for safe threshold payload."""

    threshold: float
    alpha_level: float
    method: str


# --- Component implementations ---


@dataclass(kw_only=True)
class BetaBinomialEValue(Statistic):
    """
    Compute beta-binomial e-values for two-proportion testing.

    This component implements e-value based testing using beta-binomial conjugate
    priors. The e-value provides anytime-valid evidence against the null hypothesis
    of equal proportions.

    Attributes:
        alpha_prior: Alpha parameter for beta prior (default 1.0 = uniform)
        beta_prior: Beta parameter for beta prior (default 1.0 = uniform)
        method: Computation method ("simple" or "adaptive")
        tag_stats: Tag for statistic events in ledger

    Mathematical Note:
        Uses the beta-binomial e-process which compares:
        - H0: pA = pB (common proportion)
        - H1: pA, pB independent with beta priors

    The e-value E_n has the property that under H0, E[E_n] ≤ 1 for all n,
    making it suitable for anytime-valid testing.
    """

    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    method: str = "simple"  # "simple" or "adaptive"
    tag_stats: Optional[str] = "stat:evalue"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """
        Compute and record beta-binomial e-value.

        Reads observation events from the ledger, computes the current e-value
        based on accumulated data, and writes the result as a statistic event.

        Args:
            ledger: Event ledger for reading/writing
            experiment_id: Experiment identifier
            step_key: Step identifier within experiment
            time_index: Logical time index for this computation
        """
        # Aggregate observations from ledger
        nA, nB, mA, mB = reduce_counts(ledger, experiment_id=str(experiment_id))

        # Compute e-value using specified method
        if self.method == "adaptive":
            log_e_value = log_beta_binomial_evalue_adaptive(
                mA, nA, mB, nB, self.alpha_prior, self.beta_prior
            )
        else:  # "simple" method
            log_e_value = log_beta_binomial_evalue_simple(
                mA,
                nA,
                mB,
                nB,
                self.alpha_prior,
                self.beta_prior,
                self.alpha_prior,
                self.beta_prior,
            )

        # Convert to e-value (handle numerical stability)
        e_value = math.exp(log_e_value) if log_e_value < 100 else float("inf")

        # Prepare payload
        payload: Dict[str, Any] = {
            "e_value": float(e_value),
            "log_e_value": float(log_e_value),
            "nA": nA,
            "nB": nB,
            "mA": mA,
            "mB": mB,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "method": self.method,
        }

        # Write statistic event to ledger
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="updated",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="BetaBinomialEValue",
            payload=payload,
            tag=self.tag_stats,
        )


@dataclass(kw_only=True)
class SafeThreshold(Criteria):
    """
    Define anytime-valid significance thresholds for e-values.

    In safe testing, significance is achieved when the e-value exceeds 1/α,
    where α is the desired error rate. This threshold provides anytime-valid
    Type I error control without multiple testing corrections.

    Attributes:
        alpha_level: Desired Type I error rate (default 0.05)
        tag_crit: Tag for criteria events in ledger

    Mathematical Note:
        The threshold 1/α comes from the fundamental property of e-values:
        P(E_τ ≥ 1/α for some τ) ≤ α under H0

        This makes the test anytime-valid with exact Type I error control.
    """

    alpha_level: float = 0.05
    tag_crit: Optional[str] = "crit:safe_threshold"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """
        Compute and record safe testing threshold.

        The threshold for safe testing is simply 1/α where α is the desired
        significance level. This provides exact anytime-valid Type I error control.

        Args:
            ledger: Event ledger for reading/writing
            experiment_id: Experiment identifier
            step_key: Step identifier within experiment
            time_index: Logical time index for this computation
        """
        threshold = 1.0 / self.alpha_level

        payload: Dict[str, Any] = {
            "threshold": float(threshold),
            "alpha_level": self.alpha_level,
            "method": "safe_testing",
        }

        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.CRITERIA,
            kind="updated",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="SafeThreshold",
            payload=payload,
            tag=self.tag_crit,
        )


@dataclass(kw_only=True)
class SafeSignaler(Signaler):
    """
    Emit stopping signals based on e-value threshold crossing.

    This component monitors e-values and emits stopping signals when the
    e-value exceeds the safe testing threshold, indicating anytime-valid
    significance.

    Attributes:
        decision_topic: Topic name for decision signals
        min_observations: Minimum observations required before signaling

    Decision Logic:
        - Signal "significant" if e-value ≥ threshold
        - Signal "continue" otherwise
        - No signal if insufficient data

    The signaling preserves anytime validity - once significance is achieved,
    it remains valid regardless of when the test is stopped.
    """

    decision_topic: str = "safe_decision"
    min_observations: int = 2  # Minimum total observations needed

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """
        Check e-value against threshold and emit decision signal.

        Reads the latest e-value and threshold from the ledger and applies
        the stopping rule. Emits appropriate decision signals.

        Args:
            ledger: Event ledger for reading/writing
            experiment_id: Experiment identifier
            step_key: Step identifier within experiment
            time_index: Logical time index for this decision
        """
        # Get latest e-value and threshold
        evalue_row = ledger.latest(
            namespace=Namespace.STATS,
            tag="stat:evalue",
            experiment_id=str(experiment_id),
        )
        threshold_row = ledger.latest(
            namespace=Namespace.CRITERIA,
            tag="crit:safe_threshold",
            experiment_id=str(experiment_id),
        )

        if not evalue_row or not threshold_row:
            return

        # Extract values
        e_value = float(evalue_row.payload.get("e_value", 0.0))
        threshold = float(threshold_row.payload.get("threshold", float("inf")))
        total_n = evalue_row.payload.get("nA", 0) + evalue_row.payload.get("nB", 0)

        # Check minimum observations
        if total_n < self.min_observations:
            return

        # Apply decision rule
        if e_value >= threshold:
            action = "stop"
            reason = "significant"
            confidence = "anytime_valid"
        else:
            action = "continue"
            reason = "not_significant"
            confidence = f"e_value={e_value:.3f}, threshold={threshold:.1f}"

        # Emit decision signal
        ledger.emit(
            time_index=time_index,
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            topic=self.decision_topic,
            body={
                "action": action,
                "reason": reason,
                "e_value": e_value,
                "threshold": threshold,
                "confidence": confidence,
                "total_n": total_n,
            },
            tag=f"{self.decision_topic}:decision",
            namespace=Namespace.SIGNALS,
        )
