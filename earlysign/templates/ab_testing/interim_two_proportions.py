"""
ABTesting: Interim Analysis (Group Sequential) for two-proportions.

Pipeline
--------
counts -> information-time -> design -> boundary -> statistic -> decision
(with optional gate before running the test)

This template is a thin facade that composes scheme-agnostic GS blocks with
two-proportions statistics. It keeps the ledger as the source of truth.
"""

from typing import Any, Dict, List, Mapping, Optional, Union

from earlysign.framework.gate import GateDecisionRecord
from earlysign.framework.templates import AnalysisResult, TemplateBase
from earlysign.reporting.bridge import ReportingBridgeMixin
from earlysign.stats.common.group_sequential.operators.boundary_op import (
    BoundaryFromDesign,
)
from earlysign.stats.common.group_sequential.operators.design_op import (
    GroupSequentialDesign,
)
from earlysign.stats.common.group_sequential.operators.info_op import (
    InformationTime,
    InformationTimeFromFisher,
    InformationTimeFromRatio,
    InformationTimeFromSD,
    InformationTimeFromVariance,
)
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDesignRecord,
    InformationTimeRecord,
)
from earlysign.stats.schemes.two_proportions.gates import GateMinTotalSamples
from earlysign.stats.schemes.two_proportions.group_sequential import GSDecisionFromWaldZ
from earlysign.stats.schemes.two_proportions.operators import WaldZStatistic
from earlysign.stats.schemes.two_proportions.records import (
    BinomialCountsRecord,
    WaldZStatisticRecord,
)


class InterimAnalysisTwoProportions(ReportingBridgeMixin, TemplateBase):
    """
    Group Sequential A/B test for binary outcomes (A vs B).

    Usage sketch
    ------------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> ledger = Ledger(ibis.duckdb.connect(":memory:"), "table")
    >>> ledger.ensure()
    >>> t = InterimAnalysisTwoProportions(experiment_id="exp1", namespace="tp")
    >>> t.setup(ledger.bind(experiment_id="exp1"))
    >>> t.design_interim(alpha=0.05, tails=2, style="alpha_spending", family="obf", scale="z",
    ...                  futility={"mode": "symmetric"})
    >>> t.set_info_plan(kind="counts", planned_max_n=220)
    >>> t.add_observations(nA=100, mA=40, nB=120, mB=55, look=1)
    >>>
    >>> res = t.analyze()
    """

    # ------------------------------ lifecycle ------------------------------

    def build_components(self) -> None:
        ids = {
            "counts": self.mkid("counts"),
            "info": self.mkid("info"),
            "design": self.mkid("design"),
            "boundary": self.mkid("boundary"),
            "wald": self.mkid("wald"),
            "decision": self.mkid("decision"),
            "gate": self.mkid("gate"),
        }
        self.registry["ids"] = ids
        # defaults: users can override via design_interim / set_info_plan / set_gate
        self.registry["params"] = {
            "info_plan": {
                "kind": "counts",
                "planned_max_n": None,
                "planned_fractions": None,
            },
            "gate": {"enabled": False, "min_total": 0},
        }
        self.registry["design"] = {}

    # ------------------------------ design --------------------------------

    def design_interim(
        self,
        *,
        alpha: float,
        tails: int,
        style: str,
        scale: str = "z",
        family: Optional[str] = None,
        alpha_levels: Optional[Union[Mapping[int, float], List[float]]] = None,
        futility: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a GS design row.

        Parameters follow `earlysign.stats.common.group_sequential.design`.
        """
        ids = self.registry["ids"]
        design_payload: Dict[str, Any] = {
            "alpha": float(alpha),
            "tails": int(tails),
            "scale": str(scale),
            "efficacy": {"style": str(style)},
        }
        if family is not None:
            design_payload["efficacy"]["family"] = str(family)
        if alpha_levels is not None:
            design_payload["efficacy"]["alpha_levels"] = alpha_levels
        if futility is not None:
            design_payload["futility"] = dict(futility)

        # persist
        GroupSequentialDesign(
            self.scoped,
            out_id=ids["design"],
            design=design_payload,
        ).run()
        self.registry["design"] = design_payload

    def set_info_plan(self, *, kind: str = "counts", **kwargs: Any) -> None:
        """Configure information-time estimation plan for analyze().

        Options
        -------
        kind="counts": needs planned_max_n=int
        kind="ratio" : info_now, info_max
        kind="variance": var_now, var_target
        kind="sd": sd_now, sd_target
        kind="fisher": fisher_now, fisher_max
        """
        self.registry["params"]["info_plan"] = {"kind": str(kind), **kwargs}

    def set_gate(self, *, enabled: bool, min_total: int) -> None:
        """Enable/disable and configure the min-samples gate."""
        self.registry["params"]["gate"] = {
            "enabled": bool(enabled),
            "min_total": int(min_total),
        }

    # ------------------------------ data IO -------------------------------

    def add_observations(
        self, *, nA: int, mA: int, nB: int, mB: int, look: Optional[int] = None
    ) -> None:
        """Insert a counts snapshot. `look` is optional."""
        ids = self.registry["ids"]
        counts = BinomialCountsRecord(id=ids["counts"]).attach(self.scoped)
        payload = {"nA": int(nA), "mA": int(mA), "nB": int(nB), "mB": int(mB)}
        if look is not None:
            payload["look"] = int(look)
        counts.insert(payload)

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
        # Information time
        info_plan = params["info_plan"]
        kind = info_plan.get("kind", "counts")
        if kind == "counts":
            counts_rec = BinomialCountsRecord(id=ids["counts"]).attach(self.scoped)
            planned_max_n = info_plan.get("planned_max_n")
            if planned_max_n is None:
                raise ValueError(
                    "planned_max_n must be provided in info_plan for kind='counts'"
                )
            InformationTime(
                self.scoped,
                out_id=ids["info"],
                cum_counts=counts_rec,
                planned_max_n=planned_max_n,
            ).run()
        elif kind == "ratio":
            InformationTimeFromRatio(
                self.scoped,
                out_id=ids["info"],
                info_now=info_plan["info_now"],
                info_max=info_plan["info_max"],
            ).run()
        elif kind == "variance":
            InformationTimeFromVariance(
                self.scoped,
                out_id=ids["info"],
                var_now=info_plan["var_now"],
                var_target=info_plan["var_target"],
            ).run()
        elif kind == "sd":
            InformationTimeFromSD(
                self.scoped,
                out_id=ids["info"],
                sd_now=info_plan["sd_now"],
                sd_target=info_plan["sd_target"],
            ).run()
        elif kind == "fisher":
            InformationTimeFromFisher(
                self.scoped,
                out_id=ids["info"],
                fisher_now=info_plan["fisher_now"],
                fisher_max=info_plan["fisher_max"],
            ).run()
        else:
            raise ValueError("Unknown info plan kind: {}".format(kind))

        # Boundary from design
        BoundaryFromDesign(
            self.scoped,
            design=GroupSequentialDesignRecord(id=ids["design"]).attach(self.scoped),
            info=InformationTimeRecord(id=ids["info"]).attach(self.scoped),
            out_id=ids["boundary"],
            look=None,
        ).run()

        # Statistic
        WaldZStatistic(
            self.scoped,
            cum_counts=BinomialCountsRecord(id=ids["counts"]).attach(self.scoped),
            out_id=ids["wald"],
            pooled=True,
        ).run()

        # Decision (scale-aware)
        GSDecisionFromWaldZ(
            self.scoped,
            wald=WaldZStatisticRecord(id=ids["wald"]).attach(self.scoped),
            boundary=GroupSequentialBoundaryRecord(id=ids["boundary"]).attach(
                self.scoped
            ),
            out_id=ids["decision"],
            value_scale="z",
            info=InformationTimeRecord(id=ids["info"]).attach(self.scoped),
        ).run()

        # Compose result
        GroupSequentialBoundaryRecord(id=ids["boundary"]).attach(self.scoped)

        ddf = (
            GroupSequentialBoundaryRecord(id=ids["boundary"])
            .attach(self.scoped)
            .latest()
            .select("upper", "lower", "scale")
            .execute()
            .iloc[0]
        )
        sdf = (
            WaldZStatisticRecord(id=ids["wald"])
            .attach(self.scoped)
            .latest()
            .select(
                wz=WaldZStatisticRecord(id=ids["wald"])
                .attach(self.scoped)
                .t.payload["wald_z"]
                .cast("float64")
            )
            .execute()
        )
        (
            GroupSequentialBoundaryRecord(id=ids["boundary"])
            .attach(self.scoped)  # decision record lives elsewhere, so fetch directly
            .latest()
            .execute()
        )
        # decision row
        (
            GroupSequentialBoundaryRecord(id=ids["boundary"])
            .attach(self.scoped)
            .latest()
            .execute()
        )
        # read decision
        from earlysign.stats.common.group_sequential.records import (
            GroupSequentialDecisionSignalRecord,
        )

        ddf2 = (
            GroupSequentialDecisionSignalRecord(id=ids["decision"])
            .attach(self.scoped)
            .latest()
            .select(
                signal=GroupSequentialDecisionSignalRecord(id=ids["decision"])
                .attach(self.scoped)
                .t.payload["signal"],
                reason=GroupSequentialDecisionSignalRecord(id=ids["decision"])
                .attach(self.scoped)
                .t.payload["reason"],
            )
            .execute()
        )
        if len(ddf2) == 0:
            return AnalysisResult(
                signal="continue",
                reason="no_decision",
                stopped=False,
                summary={},
                raw={},
            )

        signal = str(ddf2.iloc[0]["signal"])
        reason = str(ddf2.iloc[0]["reason"])
        stopped = signal in ("stop_efficacy", "stop_futility")

        summary: Dict[str, Any] = {"signal": signal, "reason": reason}
        summary["boundary"] = {
            "upper": float(ddf["upper"]),
            "lower": float(ddf["lower"]),
            "scale": str(ddf["scale"]),
        }
        if len(sdf) > 0:
            summary["wald_z"] = float(sdf.iloc[0]["wz"])

        raw = {"ids": ids}
        return AnalysisResult(
            signal=signal, reason=reason, stopped=stopped, summary=summary, raw=raw
        )
