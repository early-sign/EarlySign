from typing import TYPE_CHECKING

from pydantic import BaseModel

from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.write_models import WriteModel
from earlysign.v1.methods.actions import Decision, UpdateProtocol
from earlysign.v1.methods.anytime_valid.e_process import (
    EValueResult,
    compute_binomial_e_value,
)
from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol
from earlysign.v1.stats.binary import BinomialSummaryFact

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger
    from earlysign.v1.framework.trace import TraceHash


class DecisionRecord(BaseModel):
    action: str
    e_value: float


class BinomialMonitoringTemplate:
    """
    Safe testing / Continuous monitoring template using e-processes (Pattern G).
    """

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    def set_protocol(self, protocol: EProcessProtocol) -> "TraceHash":
        with Session(self.ledger) as sess:
            return UpdateProtocol(sess, protocol)

    def check(self) -> EValueResult:
        """
        Performs an e-check against the current ledger state using the persisted protocol.
        """
        with Session(self.ledger) as sess:
            # 1. Read Protocol from Ledger (Scientific Intent)
            protocol_traced = sess.Read(ProtocolProjector(EProcessProtocol))
            protocol = protocol_traced.data

            # 2. Read Summary (Scientific Evidence)
            proj = BinomialSummaryFact(identity="monitoring_summary")
            traced_summary = sess.Read(proj)

            # NOTE: We no longer manually call proj.save() here.
            # Snapshots are optimizations that can be performed during ingestion
            # or at specific milestones, but shouldn't be forced during every check.

            # 3. Compute e-value via CommitCallResult (Scientific Provenance)
            if protocol.alt_p is None:
                raise ValueError(
                    "EProcessProtocol must specify alt_p for Binomial Monitoring"
                )

            res = WriteModel.CommitCallResult(
                sess,
                EValueResult,
                compute_binomial_e_value,
                n=traced_summary.data.n,
                successes=traced_summary.data.successes,
                null_p=protocol.null_p,
                alt_p=protocol.alt_p,
                alpha=protocol.alpha,
            )

            # 4. Decision
            if res.is_rejected:
                Decision(
                    sess, DecisionRecord(action="Reject H0", e_value=float(res.e_value))
                )

            return res
