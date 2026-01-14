import json
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    cast,
)

import ibis
from pydantic import BaseModel

from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.framework.trace import Traced
from earlysign.v1.framework.write_models import WriteModel

if TYPE_CHECKING:
    from earlysign.v1.framework.session import Session

T = TypeVar("T", bound=BaseModel)


class Snapshot(BaseModel, Generic[T]):
    """
    A recomputable intermediate fact (Memento).
    """

    identity: str
    data: T
    ts: Any  # Ledger's Last Timestamp
    uuid: Optional[str] = None  # Record uuid for trace reference


class IntermediateFact(Projector[T], ABC):
    """
    Base class for recomputable intermediate facts using Incremental Folding.
    """

    data_type: Type[T]

    def __init__(self, identity: str):
        self.identity = identity

    @abstractmethod
    def compute(
        self,
        snapshot: Optional[Snapshot[T]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[T]:
        """
        Implemented by subclasses to perform the actual folding.
        """
        pass

    def project(self, table: ibis.Expr) -> ProjectionResult[T]:
        """
        Coordinates the reconstruction of state from the Ledger.
        """
        # 1. Retrieve latest snapshot for this identity
        snapshot = self._find_latest_snapshot(table)

        # 2. Filter for delta (new events since snapshot)
        if snapshot:
            delta_expr = table.filter(table.ts > snapshot.ts)
        else:
            delta_expr = table

        # 3. Delegate realization
        return self.compute(snapshot, delta_expr, table)

    def _find_latest_snapshot(self, table: ibis.Expr) -> Optional[Snapshot[T]]:
        """
        Efficiently find the latest snapshot for this identity.
        """
        # We search specifically for snapshots of this identity
        # The identity column is now top-level in the ledger
        snaps = table.filter(table.payload_type == "Snapshot")
        matched = snaps.filter(snaps.identity == self.identity)

        # Get the latest one
        latest = matched.order_by(ibis.desc("ts")).limit(1).execute()

        if not latest.empty:
            row = latest.iloc[0]

            # Robust Metadata Parsing
            def _ensure_dict(val: Any) -> Dict[str, Any]:
                if isinstance(val, str):
                    return cast(Dict[str, Any], json.loads(val))
                return dict(val) if val is not None else {}

            payload = _ensure_dict(row.get("payload"))
            data_raw = payload.get("data")

            if data_raw is None:
                raise KeyError(
                    f"Snapshot for {self.identity} is missing 'data' in payload."
                )

            # Hydrate the data into the expected Pydantic model T
            try:
                data_inst = (
                    self.data_type(**data_raw)
                    if isinstance(data_raw, dict)
                    else data_raw
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to hydrate snapshot data for {self.identity}: {e}"
                ) from e

            return Snapshot(
                identity=self.identity,
                data=data_inst,
                ts=row["ts"],
                uuid=row.get("uuid"),
            )
        return None

    def save(self, session: "Session", result: Traced[T]) -> None:
        """
        Standardizes how a new intermediate fact is committed.
        """
        # Create a Snapshot record
        # Note: We include the session.horizon_id as the 'ts' anchor
        snap_record = Snapshot(
            identity=self.identity, data=result.data, ts=session.horizon_id
        )

        # We use a custom Commit that ensures identity is set in labels
        WriteModel.Commit(session, snap_record, labels={"identity": self.identity})
