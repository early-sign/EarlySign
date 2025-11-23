"""Records for the two-proportions scheme."""

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp
from earlysign.framework.records import LedgerRecord, QueryMixin, SnapshotLedgerRecord


class BinomialArmResultRecord(LedgerRecord, QueryMixin):
    """Incremental observation for a single arm."""

    schema = {
        "trial": (int, ...),
        "success": (int, ...),
    }


class BinomialArmSnapshot(SnapshotLedgerRecord, LedgerOp):  # type: ignore[misc]
    """
    Cumulative totals for a single arm.

    When initialised with ``obs`` and ``arm_name`` it can be used as an operator:
    ``run()`` will read the latest observation delta and append an updated snapshot.
    """

    schema = {
        "trial": (int, ...),
        "success": (int, ...),
    }
    snapshot_of = BinomialArmResultRecord

    def __init__(
        self,
        name: str,
        *,
        ledger: Ledger | None = None,
        obs: BinomialArmResultRecord | None = None,
        arm_name: str | None = None,
    ):
        super().__init__(name, ledger=ledger, snapshot_of=BinomialArmResultRecord)
        self.obs = obs
        self.arm_name = arm_name

    def run(self) -> None:
        obs = self.obs
        if obs is None:
            raise RuntimeError("BinomialArmSnapshot.run() requires 'obs'.")

        arm_name = self.arm_name
        if arm_name is None:
            raise RuntimeError("BinomialArmSnapshot.run() requires 'arm_name'.")

        ledger = self.ledger or obs.ledger
        if ledger is None:
            raise RuntimeError("Ledger must be attached before running snapshot.")
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


class ScoreZStatisticRecord(LedgerRecord, QueryMixin):
    """Score Z statistic record for two-proportions."""

    schema = {
        "score_z": (float, ...),
    }
