"""
Snapshot model for entity state persistence.
"""

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
"""Generic type placeholder for the snapshotted data"""


class Snapshot(BaseModel, Generic[T]):
    """A recomputable intermediate fact (Memento).

    Snapshots cache the state of an Entity at a given point in time.
    They can always be recomputed from the underlying events.

    Attributes:
        entity_identity: The unique identity of the entity.
        data: The captured state data.
        timestamp: The ledger's last timestamp at the time of snapshot.
        uuid: Optional record ID for trace reference.
    """

    entity_identity: str
    data: T
    timestamp: Any  # Ledger's Last Timestamp
    uuid: Optional[str] = None  # Record record_id for trace reference
