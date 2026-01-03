from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

from earlysign.v1.framework.session import Session
from earlysign.v1.methods.actions import Decision, Ingest, UpdateProtocol
from earlysign.v1.methods.group_sequential.protocol import GSTProtocol
from earlysign.v1.methods.binomial import BinomialSummaryFact
from earlysign.v1.methods.group_sequential.binomial import (
    BinomialZProjector,
)
from earlysign.v1.methods.group_sequential.report import (
    BinomialFinalProjector,
    BinomialProgressProjector,
)

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger
    from earlysign.v1.framework.trace import TraceHash


class ABDecisionRecord(BaseModel):
    status: str
    message: str


class BinomialABTemplate:
    """
    Standard orchestration for a Binomial A/B test using Group Sequential Design (Pattern G).
    """

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    def set_protocol(self, protocol: GSTProtocol):
        """
        Persists the trial protocol to the ledger.
        This handles both initial intent and realized designs.
        """
        with Session(self.ledger) as sess:
            UpdateProtocol(sess, protocol)

    def update(self, batch: List[BaseModel]) -> Dict[str, Any]:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        from earlysign.v1.framework.projector import ProtocolProjector

        # 1. Ingest Evidence
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    Ingest(sess, item)

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            p = sess.Read(ProtocolProjector(GSTProtocol)).data

            summary_c = sess.Read(
                BinomialSummaryFact(identity="summary_c", filter_variant="C")
            ).data
            summary_t = sess.Read(
                BinomialSummaryFact(identity="summary_t", filter_variant="T")
            ).data

            cumulative_n = summary_c.n + summary_t.n
            info_frac = cumulative_n / p.n_max if p.n_max > 0 else 0

            result = {
                "info_frac": info_frac,
                "look": None,
                "z_stat": None,
                "is_rejected": False,
                "status": "CONTINUE",
            }

            # 3. Check if Look is due
            # We determine the "current" look by comparing cumulative N with milestones.
            look_num = None
            boundary = None
            for i, m in enumerate(p.milestones):
                if info_frac >= m:
                    look_num = i + 1
                    boundary = p.boundaries[i]

            if look_num:
                # Execution layer: Pure statistical calculation
                analysis_traced = sess.Read(BinomialZProjector(boundary))
                calc_res = analysis_traced.data

                # Record Decision if rejected or final
                status = "CONTINUE"
                if calc_res.is_rejected:
                    status = "STOP_EFFICACY"
                    Decision(
                        sess,
                        ABDecisionRecord(
                            status="STOP", message=f"Rejected at Look {look_num}"
                        ),
                        trace=analysis_traced.trace,
                    )
                elif look_num == len(p.milestones):
                    status = "STOP_FINAL"

                result.update(
                    {
                        "look": look_num,
                        "z_stat": calc_res.z_stat,
                        "is_rejected": calc_res.is_rejected,
                        "status": status,
                    }
                )

            return result

    def progress_report(self) -> Dict[str, Any]:
        """
        Returns the current progress report.
        Reconstructs state via BinomialProgressProjector.
        """
        with Session(self.ledger) as sess:
            return sess.Read(BinomialProgressProjector()).data.model_dump()

    def final_report(self, is_rejected: bool, final_status: str) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.Read(
                BinomialFinalProjector(is_rejected=is_rejected, final_status=final_status)
            ).data.model_dump()

    def run_backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
        Returns a FinalReport.

        Data Requirements:
        - `batches`: Iterator yielding `BatchObservation` objects or lists of them.
        - Each `BatchObservation` must have `n`, `success`, and `variant`.
        """
        last_res = {"status": "COMPLETED", "is_rejected": False}
        for i, batch in enumerate(batches):
            res = self.update(batch if isinstance(batch, list) else [batch])
            last_res = res
            if res.get("status") == "STOP_EFFICACY":
                return self.final_report(is_rejected=True, final_status="STOP_EFFICACY")

        return self.final_report(
            is_rejected=last_res.get("is_rejected", False),
            final_status=last_res.get("status", "COMPLETED"),
        )
