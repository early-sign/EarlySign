"""
earlysign.stats.schemes.two_proportions.e_process
================================================

Safe testing and e-process components for two-proportions testing.

This module applies the generic e-process and safe testing methods from
earlysign.stats.common.e_process to the specific case of comparing two
binomial proportions using beta-binomial e-values.

Components:
- BetaBinomialEValue: Compute beta-binomial e-values from ledger observations
- SafeThreshold: Define anytime-valid thresholds for two-proportions
- SafeSignaler: Safe testing decision signaling for two-proportions
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Union, Dict, Any

from earlysign.core.components import Criteria, Signaler, Statistic
from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.core.ledger import Ledger
from earlysign.stats.common.e_process import (
    log_beta_binomial_evalue_simple,
    log_beta_binomial_evalue_adaptive,
    safe_threshold,
)
from earlysign.stats.schemes.two_proportions.common import (
    reduce_counts,
    get_latest_statistic,
    get_latest_criteria,
)


# --- Safe Testing Components ---


@dataclass(kw_only=True)
class BetaBinomialEValue(Statistic):
    """
    Compute beta-binomial e-values for two-proportions testing.

    Applies the generic beta-binomial e-value computation to two-proportions
    data from the ledger. Implements e-value based testing using beta-binomial
    conjugate priors for anytime-valid evidence against the null hypothesis
    of equal proportions.

    Mathematical background:
        Tests H0: pA = pB vs H1: pA, pB independent with beta priors
        E-value has the property E[E_n] ≤ 1 under H0 for all n

    Attributes:
        alpha_prior: Alpha parameter for beta prior (default 1.0 = uniform)
        beta_prior: Beta parameter for beta prior (default 1.0 = uniform)
        method: Computation method ("simple" or "adaptive")
        tag_stats: Tag for statistic events

    Events consumed:
        - Namespace.OBS: TwoPropObsBatch observations

    Events produced:
        - Namespace.STATS: BetaBinomialEValue with e-value and metadata
    """

    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    method: str = "simple"  # "simple" or "adaptive"
    tag_stats: str = "stat:evalue"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """
        Compute and record beta-binomial e-value for two-proportions.

        Reads observation events from the ledger, computes the current e-value
        based on accumulated two-proportions data, and writes the result as a
        statistic event.
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

        # Prepare payload with two-proportions specific info
        payload: Dict[str, Any] = {
            "e_value": float(e_value),
            "log_e_value": float(log_e_value),
            "nA": nA,
            "nB": nB,
            "mA": mA,
            "mB": mB,
            "pA_hat": mA / max(nA, 1),
            "pB_hat": mB / max(nB, 1),
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "method": self.method,
            "scheme": "two_proportions",
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
    Define anytime-valid significance thresholds for two-proportions e-values.

    Applies the generic safe testing threshold to two-proportions testing.
    In safe testing, significance is achieved when the e-value exceeds 1/α,
    where α is the desired error rate.

    Mathematical foundation:
        Threshold = 1/α provides exact Type I error control:
        P(E_τ ≥ 1/α for some τ) ≤ α under H0

    Attributes:
        alpha_level: Desired Type I error rate (default 0.05)
        tag_crit: Tag for criteria events in ledger

    Events produced:
        - Namespace.CRITERIA: SafeThreshold with threshold value
    """

    alpha_level: float = 0.05
    tag_crit: str = "crit:safe_threshold"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Compute and record safe testing threshold for two-proportions."""
        threshold = safe_threshold(self.alpha_level)

        payload: Dict[str, Any] = {
            "threshold": float(threshold),
            "alpha_level": self.alpha_level,
            "method": "safe_testing",
            "scheme": "two_proportions",
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
    Emit stopping signals based on e-value threshold crossing for two-proportions.

    Monitors e-values and emits stopping signals when the e-value exceeds
    the safe testing threshold, indicating anytime-valid significance for
    two-proportions comparisons.

    Decision logic:
        - Signal "stop" with "significant" reason if e-value ≥ threshold
        - Signal "continue" with "not_significant" reason otherwise
        - No signal if insufficient data

    Attributes:
        decision_topic: Topic name for decision signals
        min_observations: Minimum observations required before signaling

    Events consumed:
        - Namespace.STATS: BetaBinomialEValue events
        - Namespace.CRITERIA: SafeThreshold events

    Events produced:
        - Namespace.SIGNALS: Decision signals with action and evidence
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
        """Check e-value against threshold and emit decision signal for two-proportions."""
        # Get latest e-value and threshold
        evalue_payload = get_latest_statistic(ledger, str(experiment_id), "stat:evalue")
        threshold_payload = get_latest_criteria(
            ledger, str(experiment_id), "crit:safe_threshold"
        )

        if not evalue_payload or not threshold_payload:
            return

        # Extract values
        e_value = float(evalue_payload.get("e_value", 0.0))
        threshold = float(threshold_payload.get("threshold", float("inf")))
        total_n = evalue_payload.get("nA", 0) + evalue_payload.get("nB", 0)

        # Check minimum observations
        if total_n < self.min_observations:
            return

        # Extract two-proportions specific information
        pA_hat = evalue_payload.get("pA_hat", 0.0)
        pB_hat = evalue_payload.get("pB_hat", 0.0)
        nA = evalue_payload.get("nA", 0)
        nB = evalue_payload.get("nB", 0)

        # Apply decision rule
        if e_value >= threshold:
            action = "stop"
            if pB_hat > pA_hat:
                reason = "significant_positive"  # B better than A
                direction = "positive"
            else:
                reason = "significant_negative"  # A better than B
                direction = "negative"
            confidence = "anytime_valid"
        else:
            action = "continue"
            reason = "not_significant"
            direction = "none"
            confidence = f"e_value={e_value:.3f}, threshold={threshold:.1f}"

        # Emit decision signal with two-proportions context
        ledger.write_event(
            time_index=time_index,
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            namespace=Namespace.SIGNALS,
            kind="decision",
            tag=f"{self.decision_topic}:decision",
            payload_type="dict",
            payload={
                "action": action,
                "reason": reason,
                "direction": direction,
                "e_value": e_value,
                "threshold": threshold,
                "confidence": confidence,
                "total_n": total_n,
                "nA": nA,
                "nB": nB,
                "pA_hat": pA_hat,
                "pB_hat": pB_hat,
                "effect_size": pB_hat - pA_hat,
                "scheme": "two_proportions",
            },
        )
