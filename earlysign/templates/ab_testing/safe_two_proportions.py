"""
ABTesting: Safe Testing (Anytime-Valid, Ville) for two-proportions.

Pipeline
--------
counts -> e-process -> ville-threshold -> decision
(optional gate may be applied before the test)
"""

from typing import Any, Dict, Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.templates import TemplateBase, AnalysisResult
from earlysign.framework.gate import GateDecisionRecord
from earlysign.stats.schemes.two_proportions.records import BinomialCountsRecord
from earlysign.stats.schemes.two_proportions.gates import GateMinTotalSamples

from earlysign.stats.common.anytime_valid.records import (
    EProcessRecord,
    VilleThresholdRecord,
    SafeDecisionRecord,
)
from earlysign.stats.schemes.two_proportions.anytime_valid import (
    MixtureEProcessTwoProportions,
)
from earlysign.stats.common.anytime_valid.ville import (
    VilleThreshold,
    VilleDecision,
)


class SafeTestingTwoProportions(TemplateBase):
    """
    Anytime-Valid A/B test for binary outcomes (A vs B) via Ville's inequality.

    Usage sketch
    ------------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> ledger = Ledger(ibis.duckdb.connect(":memory:"), "table")
    >>> ledger.ensure()
    >>> t = SafeTestingTwoProportions(experiment_id="exp2", namespace="tp")
    >>> t.setup(ledger.bind(experiment_id="exp2"))
    >>> t.design_safe(alpha=0.05)  # writes threshold=20.0
    >>> t.add_observations(nA=100, mA=38, nB=120, mB=70)
    >>> res = t.analyze()
    >>> res.signal in ("reject", "stop_futility", "continue")
    True
    """

    def build_components(self) -> None:
        ids = {
            "counts": self.mkid("counts"),
            "eproc": self.mkid("eproc"),
            "threshold": self.mkid("threshold"),
            "decision": self.mkid("decision"),
            "gate": self.mkid("gate"),
        }
        self.registry["ids"] = ids
        self.registry["params"] = {
            "priors": {"null": (0.5, 0.5), "alt": (0.5, 0.5)},
            "gate": {"enabled": False, "min_total": 0},
            "alpha": 0.05,
            "futility": {"mode": "fixed", "tau": None},
        }

    # ------------------------------ design --------------------------------

    def design_safe(self, *, alpha: float) -> None:
        """Insert a Ville threshold row (1/alpha)."""
        ids = self.registry["ids"]
        VilleThreshold(self.scoped, out_id=ids["threshold"], alpha=float(alpha)).run()
        self.registry["params"]["alpha"] = float(alpha)

    def set_priors(
        self, *, null: Tuple[float, float], alt: Tuple[float, float]
    ) -> None:
        """Configure Beta priors used by the mixture e-process."""
        self.registry["params"]["priors"] = {"null": tuple(null), "alt": tuple(alt)}

    def set_gate(self, *, enabled: bool, min_total: int) -> None:
        """Enable/disable and configure the min-samples gate."""
        self.registry["params"]["gate"] = {
            "enabled": bool(enabled),
            "min_total": int(min_total),
        }

    def set_futility(self, *, mode: str = "fixed", tau: Optional[float] = None) -> None:
        """Configure futility policy for VilleDecision."""
        self.registry["params"]["futility"] = {"mode": str(mode), "tau": tau}

    # ------------------------------ data IO -------------------------------
    def add_observations(self, *, nA: int, mA: int, nB: int, mB: int) -> None:
        ids = self.registry["ids"]
        counts = BinomialCountsRecord(id=ids["counts"]).attach(self.scoped)
        counts.insert({"nA": int(nA), "mA": int(mA), "nB": int(nB), "mB": int(mB)})

    # ------------------------------ analyze -------------------------------

    def run_analysis(self) -> AnalysisResult:
        ids = self.registry["ids"]
        params = self.registry["params"]

        # Optional gate
        if params["gate"]["enabled"]:
            GateMinTotalSamples(
                self.scoped,
                counts=BinomialCountsRecord(id=ids["counts"]).attach(self.scoped),
                out_id=ids["gate"],
                min_total=params["gate"]["min_total"],
            ).run()
            gdf = (
                GateDecisionRecord(id=ids["gate"])
                .attach(self.scoped)
                .latest()
                .select(
                    gate=GateDecisionRecord(id=ids["gate"])
                    .attach(self.scoped)
                    .t.payload["gate"]
                )
                .execute()
            )
            if len(gdf) > 0 and str(gdf.iloc[0]["gate"]) == "closed":
                return AnalysisResult(
                    signal="continue",
                    reason="gate_closed",
                    stopped=False,
                    summary={"gate": "closed"},
                    raw={"gate_id": ids["gate"]},
                )

        # E-process
        pri = params["priors"]
        MixtureEProcessTwoProportions(
            self.scoped,
            counts=BinomialCountsRecord(id=ids["counts"]).attach(self.scoped),
            out_id=ids["eproc"],
            prior_null=tuple(pri["null"]),
            prior_alt=tuple(pri["alt"]),
        ).run()

        # Decision (Ville)
        fut = params["futility"]
        VilleDecision(
            self.scoped,
            eproc=EProcessRecord(id=ids["eproc"]).attach(self.scoped),
            out_id=ids["decision"],
            threshold_rec=VilleThresholdRecord(id=ids["threshold"]).attach(self.scoped),
            futility_mode=str(fut.get("mode", "none")),
            futility_tau=fut.get("tau"),
        ).run()

        # Compose result
        ddf = (
            SafeDecisionRecord(id=ids["decision"])
            .attach(self.scoped)
            .latest()
            .select(
                signal=SafeDecisionRecord(id=ids["decision"])
                .attach(self.scoped)
                .t.payload["signal"],
                reason=SafeDecisionRecord(id=ids["decision"])
                .attach(self.scoped)
                .t.payload["reason"],
                E=SafeDecisionRecord(id=ids["decision"])
                .attach(self.scoped)
                .t.payload["E"]
                .cast("float64"),
                threshold=SafeDecisionRecord(id=ids["decision"])
                .attach(self.scoped)
                .t.payload["threshold"]
                .cast("float64"),
            )
            .execute()
        )
        if len(ddf) == 0:
            return AnalysisResult(
                signal="continue",
                reason="no_decision",
                stopped=False,
                summary={},
                raw={},
            )

        signal = str(ddf.iloc[0]["signal"])
        reason = str(ddf.iloc[0]["reason"])
        stopped = signal in ("reject", "stop_futility")

        summary = {
            "signal": signal,
            "reason": reason,
            "E": float(ddf.iloc[0]["E"]),
            "threshold": float(ddf.iloc[0]["threshold"]),
        }
        raw = {"ids": ids}
        return AnalysisResult(
            signal=signal, reason=reason, stopped=stopped, summary=summary, raw=raw
        )
