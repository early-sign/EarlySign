"""Binomial arm records and snapshot utilities for two-proportion experiments."""

from earlysign.core.ledger import Ledger
from earlysign.framework.records import LedgerRecord, QueryMixin, SnapshotLedgerRecord


class BinomialArmResultRecord(LedgerRecord, QueryMixin):
    """Incremental observation for a single arm."""

    schema = {
        "trial": (int, ...),
        "success": (int, ...),
    }


class BinomialArmSnapshot(SnapshotLedgerRecord):
    """
    Cumulative totals for a single arm.

    Use ``update_from_obs`` to append a new cumulative snapshot from an observation.
    """

    schema = {
        "trial": (int, ...),
        "success": (int, ...),
    }
    snapshot_of = BinomialArmResultRecord

    def __init__(self, name: str, *, ledger: Ledger | None = None):
        super().__init__(name, ledger=ledger, snapshot_of=BinomialArmResultRecord)

    def update_from_obs(self, obs: BinomialArmResultRecord, *, arm_name: str) -> None:
        """
        Append a cumulative snapshot using the latest observation record.

        Parameters
        ----------
        obs : BinomialArmResultRecord
            Observation record containing the latest delta for this arm.
        arm_name : str
            Identifier for the arm, stored in the snapshot labels.
        """

        ledger = self.ledger or obs.ledger
        if ledger is None:
            raise RuntimeError("Ledger must be attached before updating snapshot.")
        if self.ledger is None:
            self.attach(ledger)
        elif obs.ledger is not None and obs.ledger is not self.ledger:
            raise RuntimeError("Snapshot and observation must share the same ledger.")

        delta_payload = obs.latest_payload()
        delta_trials = int(delta_payload["trial"])
        delta_successes = int(delta_payload["success"])

        prev_snapshot = self.latest_payload(default={"trial": 0, "success": 0})
        prev_trials = int(prev_snapshot["trial"])
        prev_successes = int(prev_snapshot["success"])

        self.insert(
            trial=prev_trials + delta_trials,
            success=prev_successes + delta_successes,
            labels={"arm_name": str(arm_name)},
        )

    # Backwards-compatible alias
    def run(self, obs: BinomialArmResultRecord, *, arm_name: str) -> None:
        self.update_from_obs(obs, arm_name=arm_name)
