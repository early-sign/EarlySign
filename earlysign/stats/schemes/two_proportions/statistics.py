"""
earlysign.stats.schemes.two_proportions.statistics
==================================================

Statistical components for two-proportions testing.

This module provides all statistical computation components for comparing
two binomial proportions using different methodologies:

**Group Sequential Testing (GST) Components:**
- `WaldZStatistic`: Compute Wald Z with unpooled SE from ledger counts
- `LanDeMetsBoundary`: Compute two-sided critical boundaries with error spending
- `PeekSignaler`: Emit stop signals when |Z| >= boundary

**Safe Testing Components:**
- `BetaBinomialEValue`: Compute beta-binomial e-values for two-group comparisons
- `SafeThreshold`: Define anytime-valid significance thresholds
- `SafeSignaler`: Emit stopping signals based on e-value thresholds

All components follow the event-sourcing paradigm, reading from and writing to
the ledger with immutable events. They use only the public ledger interface
and maintain complete audit trails.

Mathematical Background
-----------------------

**Group Sequential Testing:**
Uses the Wald Z-statistic with Lan-DeMets error spending functions:
    Z = (p̂_A - p̂_B) / SE_unpooled
where SE_unpooled = sqrt(p̂_A(1-p̂_A)/n_A + p̂_B(1-p̂_B)/n_B)

**Safe Testing:**
Uses beta-binomial e-processes with conjugate priors:
    E_n = ∫ ∏ Beta(y_i; α, β) ∏ Beta(z_j; α', β') dF(α,β,α',β')

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.core.names import Namespace
>>> conn = ibis.duckdb.connect(":memory:")
>>> L = Ledger(conn, "test")
>>> # Ingest observations
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":20,"nB":20,"mA":2,"mB":5})

GST workflow:
>>> WaldZStatistic().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> LanDeMetsBoundary(alpha_total=0.05, t=0.25, style="obf").step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> PeekSignaler().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP

Safe Testing workflow:
>>> BetaBinomialEValue().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> SafeThreshold().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> SafeSignaler().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any

from scipy.stats import norm  # Required for GST thresholds
from scipy.special import betaln, gammaln
from scipy.optimize import minimize_scalar

from earlysign.core.components import Criteria, Signaler, Statistic
from earlysign.core.names import (
    Namespace,
    ExperimentId,
    StepKey,
    TimeIndex,
    WaldZTag,
    GstBoundaryTag,
)
from earlysign.core.ledger import Ledger
from earlysign.stats.methods.common.statistical import wald_z_from_counts
from earlysign.stats.methods.group_sequential.core import (
    lan_demets_spending,
    nominal_alpha_increments,
)
from earlysign.stats.methods.safe_testing.core import (
    log_beta_binomial_evalue_simple,
    log_beta_binomial_evalue_adaptive,
)
from earlysign.stats.schemes.two_proportions.core import (
    reduce_counts,
    WaldZPayload,
    GstBoundaryPayload,
)


# --- Group Sequential Testing Components ---


@dataclass(kw_only=True)
class WaldZStatistic(Statistic):
    """
    Compute Wald Z-statistic for two-proportions comparison.

    Computes the unpooled Wald Z-statistic for testing H0: pA = pB:
        Z = (p̂_A - p̂_B) / SE_unpooled
    where SE_unpooled = sqrt(p̂_A(1-p̂_A)/n_A + p̂_B(1-p̂_B)/n_B)

    Events consumed:
        - Namespace.OBS: TwoPropObsBatch observations

    Events produced:
        - Namespace.STATS: WaldZ statistic with test details

    Attributes:
        tag_stats: Tag for statistic events (default "stat:waldz")
    """

    tag_stats: str = "stat:waldz"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Compute Wald Z-statistic and write to ledger."""
        nA, nB, mA, mB = reduce_counts(ledger, experiment_id=str(experiment_id))
        z, pA, pB, se = wald_z_from_counts(nA, nB, mA, mB)

        payload: WaldZPayload = {
            "z": float(z),
            "se": float(se),
            "nA": nA,
            "nB": nB,
            "mA": mA,
            "mB": mB,
            "pA_hat": pA,
            "pB_hat": pB,
        }

        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="updated",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="WaldZ",
            payload=dict(payload),
            tag=self.tag_stats,
        )


@dataclass(kw_only=True)
class LanDeMetsBoundary(Criteria):
    """
    Compute Lan-DeMets error spending boundaries for GST.

    Computes two-sided critical boundaries at information fraction t using
    the Lan-DeMets approximation for O'Brien-Fleming or Pocock spending.

    Mathematical formulation:
        α(t) = spending function at information time t
        boundary = Φ^(-1)(1 - α(t)/2)

    Attributes:
        alpha_total: Total Type I error rate to spend
        t: Information fraction (0 < t ≤ 1)
        style: Spending function style ("obf" or "pocock")
        tag_crit: Tag for criteria events
    """

    alpha_total: float
    t: float
    style: str = "obf"
    tag_crit: str = "crit:gst"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Compute GST boundary and write to ledger."""
        cum_alpha = lan_demets_spending(self.alpha_total, self.t, self.style)
        alpha_i = max(min(cum_alpha, self.alpha_total), 1e-12)
        thr = float(norm.ppf(1 - alpha_i / 2))

        payload: GstBoundaryPayload = {
            "upper": thr,
            "lower": -thr,
            "info_time": float(self.t),
            "alpha_i": alpha_i,
        }

        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.CRITERIA,
            kind="updated",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="GSTBoundary",
            payload=dict(payload),
            tag=self.tag_crit,
        )


@dataclass(kw_only=True)
class PeekSignaler(Signaler):
    """
    Emit stopping signals when |Z| exceeds GST boundary.

    Monitors the latest Wald Z-statistic and GST boundary, emitting a stop
    signal when the absolute value of Z exceeds the critical threshold.

    Decision rule:
        - Signal "stop" if |Z| ≥ boundary
        - No signal otherwise

    Attributes:
        decision_topic: Topic name for decision signals
    """

    decision_topic: str = "gst:decision"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Check boundary crossing and emit signal if needed."""
        # Get latest Z statistic
        z_query = (
            ledger.table.filter(
                (ledger.table.namespace == str(Namespace.STATS))
                & (ledger.table.tag == "stat:waldz")
                & (ledger.table.experiment_id == str(experiment_id))
            )
            .order_by(ledger.table.timestamp.desc())
            .limit(1)
        )
        z_results = z_query.execute()

        # Get latest boundary criteria
        b_query = (
            ledger.table.filter(
                (ledger.table.namespace == str(Namespace.CRITERIA))
                & (ledger.table.tag == "crit:gst")
                & (ledger.table.experiment_id == str(experiment_id))
            )
            .order_by(ledger.table.timestamp.desc())
            .limit(1)
        )
        b_results = b_query.execute()

        if z_results.empty or b_results.empty:
            return

        # Unwrap payloads
        z_records = ledger.unwrap_results(z_results)
        b_records = ledger.unwrap_results(b_results)

        z = float(z_records[0]["payload"].get("z", 0.0))
        upper = float(b_records[0]["payload"].get("upper", float("inf")))

        if abs(z) >= upper:
            ledger.write_event(
                time_index=time_index,
                experiment_id=str(experiment_id),
                step_key=str(step_key),
                namespace=Namespace.SIGNALS,
                kind="decision",
                tag="gst:decision",
                payload_type="dict",  # Default JSON payload
                payload={"action": "stop", "z": z, "threshold": upper},
            )


# --- Safe Testing Components ---


@dataclass(kw_only=True)
class BetaBinomialEValue(Statistic):
    """
    Compute beta-binomial e-values for two-proportion testing.

    Implements e-value based testing using beta-binomial conjugate priors.
    The e-value provides anytime-valid evidence against the null hypothesis
    of equal proportions without requiring sequential testing corrections.

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
        Compute and record beta-binomial e-value.

        Reads observation events from the ledger, computes the current e-value
        based on accumulated data, and writes the result as a statistic event.
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
        """Compute and record safe testing threshold."""
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

    Monitors e-values and emits stopping signals when the e-value exceeds
    the safe testing threshold, indicating anytime-valid significance.

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
        """Check e-value against threshold and emit decision signal."""
        # Get latest e-value
        evalue_query = (
            ledger.table.filter(
                (ledger.table.namespace == str(Namespace.STATS))
                & (ledger.table.tag == "stat:evalue")
                & (ledger.table.experiment_id == str(experiment_id))
            )
            .order_by(ledger.table.timestamp.desc())
            .limit(1)
        )
        evalue_results = evalue_query.execute()

        # Get latest threshold
        threshold_query = (
            ledger.table.filter(
                (ledger.table.namespace == str(Namespace.CRITERIA))
                & (ledger.table.tag == "crit:safe_threshold")
                & (ledger.table.experiment_id == str(experiment_id))
            )
            .order_by(ledger.table.timestamp.desc())
            .limit(1)
        )
        threshold_results = threshold_query.execute()

        if evalue_results.empty or threshold_results.empty:
            return

        # Unwrap payloads
        evalue_records = ledger.unwrap_results(evalue_results)
        threshold_records = ledger.unwrap_results(threshold_results)

        # Extract values
        e_value = float(evalue_records[0]["payload"].get("e_value", 0.0))
        threshold = float(
            threshold_records[0]["payload"].get("threshold", float("inf"))
        )
        total_n = evalue_records[0]["payload"].get("nA", 0) + evalue_records[0][
            "payload"
        ].get("nB", 0)

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
                "e_value": e_value,
                "threshold": threshold,
                "confidence": confidence,
                "total_n": total_n,
            },
        )
