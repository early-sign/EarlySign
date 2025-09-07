"""
earlysign.methods.group_sequential.two_proportions
==================================================

Group Sequential Testing components for two-proportions:

- `WaldZStatistic`: compute Wald Z with unpooled SE from ledger counts
- `LanDeMetsBoundary`: compute two-sided critical boundary at info fraction `t`
- `PeekSignaler`: emit stop signal when |Z| >= boundary

All components use only the public ledger trait (`LedgerOps`) and write/read
exclusively via namespaces.

Note:
- Boundary spending uses Lan–DeMets approx for OBF/Pocock.
- Doctests that require SciPy are marked `+SKIP` to keep the suite light.

Doctest (smoke of wiring only):
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> L = PolarsLedger()
>>> # ingest a small obs
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":20,"nB":20,"mA":2,"mB":5})
>>> # statistic (requires SciPy for norm)
>>> WaldZStatistic().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> # boundary at t=0.25, alpha=0.05 (OBF)
>>> LanDeMetsBoundary(alpha_total=0.05, t=0.25, style="obf").step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> # signaler
>>> PeekSignaler().step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Union, List

from scipy.stats import norm  # required for thresholds

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
from earlysign.schemes.two_proportions.reduce import reduce_counts
from earlysign.schemes.two_proportions.model import WaldZPayload, GstBoundaryPayload


# --- math helpers ---


def wald_z_from_counts(
    nA: int, nB: int, mA: int, mB: int
) -> Tuple[float, float, float, float]:
    """Return (z, pA_hat, pB_hat, se) using unpooled SE.

    Note:
        We define Z = (pB_hat - pA_hat) / SE so that a *better variant B* yields positive Z.
    """
    if min(nA, nB) == 0:
        return 0.0, 0.0, 0.0, float("inf")
    pA_hat, pB_hat = mA / max(nA, 1), mB / max(nB, 1)
    var = pA_hat * (1 - pA_hat) / max(nA, 1) + pB_hat * (1 - pB_hat) / max(nB, 1)
    se = math.sqrt(var) if var > 0 else float("inf")
    diff = pB_hat - pA_hat  # ← here (variant minus baseline)
    z = diff / se if se not in (0.0, float("inf")) else 0.0
    return z, pA_hat, pB_hat, se


def lan_demets_spending(alpha_total: float, t: float, style: str) -> float:
    """Lan–DeMets alpha spending (two-sided cumulative) for OBF/Pocock."""
    s = style.lower()
    if s in ("obf", "o'brien", "obrien", "o'brien-fleming"):
        za2 = float(norm.ppf(1 - alpha_total / 2.0))
        return 2.0 - 2.0 * float(norm.cdf(za2 / math.sqrt(max(t, 1e-12))))
    if s in ("pocock",):
        return alpha_total * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"Unknown spending style: {style}")


def nominal_alpha_increments(
    alpha_total: float, t_grid: Iterable[float], style: str
) -> List[float]:
    """Convert cumulative spending to per-look increments."""
    cum = [lan_demets_spending(alpha_total, t, style) for t in t_grid]
    inc, prev = [], 0.0
    for c in cum:
        inc.append(max(c - prev, 0.0))
        prev = c
    return inc


# --- components ---


@dataclass(kw_only=True)
class WaldZStatistic(Statistic):
    """Compute Wald Z and append to the `stats` namespace."""

    tag_stats: WaldZTag = "stat:waldz"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
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
    """Write two-sided boundary (`GSTBoundary`) for given `t` (info fraction)."""

    alpha_total: float
    t: float
    style: str = "obf"
    tag_crit: GstBoundaryTag = "crit:gst"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
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
    """Emit a stop signal when |Z| >= boundary at the current look."""

    decision_topic: str = "gst:decision"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        z_row = ledger.latest(
            namespace=Namespace.STATS,
            tag="stat:waldz",
            experiment_id=str(experiment_id),
        )
        b_row = ledger.latest(
            namespace=Namespace.CRITERIA,
            tag="crit:gst",
            experiment_id=str(experiment_id),
        )
        if not z_row or not b_row:
            return
        z = float(z_row.payload.get("z", 0.0))
        upper = float(b_row.payload.get("upper", float("inf")))
        if abs(z) >= upper:
            ledger.emit(
                time_index=time_index,
                experiment_id=str(experiment_id),
                step_key=str(step_key),
                topic=self.decision_topic,
                body={"action": "stop", "z": z, "threshold": upper},
                tag="gst:decision",
                namespace=Namespace.SIGNALS,
            )
