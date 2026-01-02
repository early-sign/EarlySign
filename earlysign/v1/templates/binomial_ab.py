from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

from earlysign.v1.framework.session import Session
from earlysign.v1.methods.actions import Decision, Ingest, UpdateProtocol
from earlysign.v1.methods.protocols import GSTProtocol
from earlysign.v1.stats.binary import (
    BinomialSummaryFact,
    BinomialZProjector,
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
        self.n_max: int = 0
        self.milestones: List[float] = []
        self.boundaries: List[float] = []
        self._next_look_idx: int = 0

    def set_protocol(self, protocol: GSTProtocol) -> "TraceHash":
        """Persists the trial protocol to the ledger."""
        with Session(self.ledger) as sess:
            return UpdateProtocol(sess, protocol)

    def update_protocol(
        self, n_max: int, milestones: List[float], boundaries: List[float]
    ) -> None:
        """Sets the computed design boundaries and look timing."""
        self.n_max = n_max
        self.milestones = milestones
        self.boundaries = boundaries
        self._next_look_idx = 0

    def update(self, batch: List[BaseModel]) -> Dict[str, Any]:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        # 1. Ingest Evidence (Outside or in its own session to advance the Horizon)
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    Ingest(sess, item)

        # 2. Analysis (New session to capture the new Horizon)
        with Session(self.ledger) as sess:
            # Check Cumulative Evidence
            summary_c = sess.Read(
                BinomialSummaryFact(identity="summary_c", filter_variant="C")
            ).data
            summary_t = sess.Read(
                BinomialSummaryFact(identity="summary_t", filter_variant="T")
            ).data

            cumulative_n = summary_c.n + summary_t.n
            info_frac = cumulative_n / self.n_max if self.n_max > 0 else 0

            result = {
                "info_frac": info_frac,
                "look": None,
                "z_stat": None,
                "is_rejected": False,
            }

            # 3. Check if Look is due
            if (
                self._next_look_idx < len(self.milestones)
                and info_frac >= self.milestones[self._next_look_idx]
            ):
                look_num = self._next_look_idx + 1
                boundary = self.boundaries[self._next_look_idx]

                # 4. realisation (Tier 2 Projection)
                analysis_traced = sess.Read(BinomialZProjector(boundary))
                calc_res = analysis_traced.data

                result.update(
                    {
                        "look": look_num,
                        "z_stat": calc_res.z_stat,
                        "is_rejected": calc_res.is_rejected,
                    }
                )

                # 6. Record Decision
                if calc_res.is_rejected:
                    Decision(
                        sess,
                        ABDecisionRecord(
                            status="STOP", message=f"Rejected at Look {look_num}"
                        ),
                    )

                self._next_look_idx += 1

            return result
