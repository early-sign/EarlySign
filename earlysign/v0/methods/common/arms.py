"""Arm lifecycle records shared across execution schemes."""

from enum import Enum

from pydantic import Field

from earlysign.v0.framework.records import LedgerRecord, QueryMixin, SnapshotLedgerRecord


class ArmStatus(str, Enum):
    """Allowed lifecycle states for an arm."""

    ACTIVE = "active"
    DROPPED = "dropped"


class ArmLifecycleRecord(LedgerRecord, QueryMixin):
    """Record lifecycle transitions (add, pause, retire) for experiment arms."""

    schema = {
        "arm_id": (str, ...),
        "status": (str, ...),  # e.g., "active", "dropped"
    }


class ArmRosterSnapshotRecord(SnapshotLedgerRecord):
    """Snapshot of active arms at a given decision point."""

    schema = {
        "active_arms": (list[str], Field(default_factory=list)),
    }
