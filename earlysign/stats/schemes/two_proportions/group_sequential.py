"""
earlysign.stats.schemes.two_proportions.group_sequential
=======================================================

Group sequential testing components for two-proportions testing.

This module applies the generic group sequential methods from
earlysign.stats.common.group_sequential to the specific case of comparing
two binomial proportions.

Components:
- WaldZStatistic: Compute Wald Z-statistic from ledger observations
- LanDeMetsBoundary: Apply Lan-DeMets spending to two-proportions testing
- PeekSignaler: Group sequential decision signaling
- AdaptiveGSTBoundary: Adaptive boundaries with information time re-estimation
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any

from earlysign.core.components import Criteria, Signaler, Statistic
from earlysign.core.ledger import Namespace
from earlysign.core.ledger import Ledger
from earlysign.stats.common.group_sequential import (
    lan_demets_spending,
    AdaptiveInfoTime,
)
from earlysign.stats.schemes.two_proportions.common import (
    reduce_counts,
    get_latest_statistic,
    get_latest_criteria,
    WaldZPayload,
    wald_z_from_counts,
)
from earlysign.stats.common.group_sequential import GstBoundaryPayload

# Import norm from scipy for boundary calculations
try:
    from scipy.stats import norm
except ImportError:
    # Fallback for testing without scipy
    class _MockNorm:
        @staticmethod
        def ppf(x: float) -> float:
            return 1.96  # Approximate for common case

    norm = _MockNorm()


# --- Group Sequential Testing Components ---


@dataclass(kw_only=True)
class WaldZStatistic(Statistic):
    """
    Compute Wald Z-statistic for two-proportions comparison.

    Applies the generic Wald Z computation to two-proportions data from
    the ledger. Computes the unpooled Wald Z-statistic for testing H0: pA = pB:
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
        experiment_id: str,
        step_key: str,
        time_index: str,
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
    Compute Lan-DeMets error spending boundaries for two-proportions GST.

    Applies the generic Lan-DeMets spending functions to two-proportions
    testing. Computes two-sided critical boundaries at information fraction t.

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
        experiment_id: str,
        step_key: str,
        time_index: str,
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
    Emit stopping signals when |Z| exceeds GST boundary for two-proportions.

    Monitors the latest Wald Z-statistic and GST boundary, emitting a stop
    signal when the absolute value of Z exceeds the critical threshold.

    Decision rule:
        - Signal "stop" if |Z| ≥ boundary
        - No signal otherwise

    Attributes:
        decision_topic: Topic name for decision signals
        min_observations: Minimum total observations before signaling
    """

    decision_topic: str = "gst:decision"
    min_observations: int = 2

    def step(
        self,
        ledger: Ledger,
        experiment_id: str,
        step_key: str,
        time_index: str,
    ) -> None:
        """Check boundary crossing and emit signal if needed."""
        # Get latest Z statistic
        z_payload = get_latest_statistic(ledger, str(experiment_id), "stat:waldz")
        b_payload = get_latest_criteria(ledger, str(experiment_id), "crit:gst")

        if not z_payload or not b_payload:
            return

        z = float(z_payload.get("z", 0.0))
        upper = float(b_payload.get("upper", float("inf")))
        total_n = z_payload.get("nA", 0) + z_payload.get("nB", 0)

        # Check minimum observations
        if total_n < self.min_observations:
            return

        if abs(z) >= upper:
            significance = "significant" if z > 0 else "significant_negative"
            ledger.write_event(
                time_index=time_index,
                experiment_id=str(experiment_id),
                step_key=str(step_key),
                namespace=Namespace.SIGNALS,
                kind="decision",
                tag="gst:decision",
                payload_type="dict",
                payload={
                    "action": "stop",
                    "reason": significance,
                    "z": z,
                    "threshold": upper,
                    "total_n": total_n,
                    "pA_hat": z_payload.get("pA_hat", 0.0),
                    "pB_hat": z_payload.get("pB_hat", 0.0),
                },
            )


@dataclass(kw_only=True)
class AdaptiveGSTBoundary(Criteria):
    """
    Adaptive group sequential boundary for two-proportions with information time re-estimation.

    This component combines LanDeMetsBoundary with AdaptiveInfoTime to provide
    boundaries that adapt to observed sample sizes rather than fixed planning.
    Applied specifically to two-proportions testing.

    Attributes:
        alpha_total: Total Type I error rate
        style: Spending function style ("obf", "pocock")
        adaptive_info_time: AdaptiveInfoTime instance for re-estimation
        tag_crit: Tag for criteria events in ledger
    """

    alpha_total: float
    style: str = "obf"
    adaptive_info_time: AdaptiveInfoTime = field(
        default_factory=lambda: AdaptiveInfoTime(initial_target=1000, looks=5)
    )
    tag_crit: str = "crit:adaptive_gst"

    def step(
        self,
        ledger: Ledger,
        experiment_id: str,
        step_key: str,
        time_index: str,
    ) -> None:
        """
        Update boundary using adaptive information time for two-proportions.

        This method:
        1. Determines current look from step_key
        2. Gets observed sample size from ledger observations
        3. Re-estimates information fraction
        4. Computes boundary using adapted information time
        """
        # Extract look number from step_key (e.g., "look_3" -> 3)
        try:
            current_look = int(step_key.replace("look_", "").replace("look-", ""))
        except (ValueError, AttributeError):
            # Fallback: assume sequential numbering
            current_look = 1

        # Get observed sample size for two-proportions
        nA, nB, _, _ = reduce_counts(ledger, experiment_id=str(experiment_id))
        observed_n = nA + nB

        # Re-estimate information fraction
        adapted_t = self.adaptive_info_time.get_info_fraction(current_look, observed_n)

        # Create boundary component with adapted information time
        boundary = LanDeMetsBoundary(
            alpha_total=self.alpha_total,
            t=adapted_t,
            style=self.style,
            tag_crit="crit:gst",
        )

        # Execute boundary step
        boundary.step(ledger, experiment_id, step_key, time_index)

        # Also log adaptation information
        planned_t = (
            self.adaptive_info_time.planned_fractions[current_look - 1]
            if current_look <= len(self.adaptive_info_time.planned_fractions)
            else 1.0
        )

        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="adapted",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="AdaptiveInfoTime",
            payload={
                "current_look": current_look,
                "observed_n": observed_n,
                "observed_nA": nA,
                "observed_nB": nB,
                "planned_t": planned_t,
                "adapted_t": adapted_t,
                "adaptation_ratio": adapted_t / planned_t if planned_t > 0 else 1.0,
                "scheme": "two_proportions",
            },
            tag="adapt:info_time",
        )
