from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel

from earlysign.v1.framework.session import Session
from earlysign.v1.methods.actions import Decision, UpdateProtocol
from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol
from earlysign.v1.methods.anytime_valid.report import (
    MonitoringFinalProjector,
    MonitoringProgressProjector,
)

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


class DecisionRecord(BaseModel):
    action: str
    e_value: float


class BinomialMonitoringTemplate:
    """
    Safe testing / Continuous monitoring template using e-processes (Pattern G).
    """

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    def set_protocol(self, protocol: EProcessProtocol):
        """
        Persists the monitoring protocol to the ledger.
        """
        with Session(self.ledger) as sess:
            UpdateProtocol(sess, protocol)

    def progress_report(self) -> Dict[str, Any]:
        """
        Performs an e-check and returns the current progress report.
        Reconstructs state via MonitoringProgressProjector.
        """
        with Session(self.ledger) as sess:
            # 1. Read Report (Projector handles protocol and summary reconstruction internally)
            traced_report = sess.Read(MonitoringProgressProjector())
            report = traced_report.data

            # 2. Record Decision if rejected
            if report.is_rejected:
                Decision(
                    sess,
                    DecisionRecord(action="Reject H0", e_value=float(report.e_value)),
                    trace=traced_report.trace,
                )

            return report.model_dump()

    def final_report(
        self, e_value: float, is_rejected: bool, final_status: str
    ) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.Read(
                MonitoringFinalProjector(
                    e_value=e_value, is_rejected=is_rejected, final_status=final_status
                )
            ).data.model_dump()

    def run_backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
        Returns a FinalReport.

        Data Requirements:
        - `batches`: Iterator yielding `BatchObservation` objects or lists of them.
        - Each `BatchObservation` must have `n`, `success`, and `variant`.
        """
        from earlysign.v1.methods.actions import Ingest

        last_e = 0.0
        for i, batch in enumerate(batches):
            with Session(self.ledger) as sess:
                items = batch if isinstance(batch, list) else [batch]
                for item in items:
                    Ingest(sess, item)

            res = self.progress_report()
            last_e = res["e_value"]
            if res["is_rejected"]:
                return self.final_report(
                    e_value=last_e, is_rejected=True, final_status="STOP_EVAL"
                )

        return self.final_report(
            e_value=last_e, is_rejected=False, final_status="COMPLETED"
        )
