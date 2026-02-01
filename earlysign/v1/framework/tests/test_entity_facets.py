from typing import Optional

import ibis
import pytest
from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.v1.framework.entity import Entity, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.trace import TraceId


# 1. Define a simple schema and entity for testing
class CounterState(BaseModel):
    count: int


class CounterEntity(Entity[CounterState]):
    data_type = CounterState

    @property
    def initial_value(self) -> CounterState:
        return CounterState(count=0)

    def compute(
        self,
        snapshot: Optional[Snapshot[CounterState]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[CounterState]:
        current_count = snapshot.data.count if snapshot else 0

        # Sum up 'Increment' events
        increments = delta_expr.filter(delta_expr.type == "Increment").execute()
        for _, row in increments.iterrows():
            current_count += row["payload"].get("value", 1)

        trace = [TraceId(str(uid)) for uid in increments["uuid"]]
        if snapshot and snapshot.uuid:
            trace.insert(0, TraceId(str(snapshot.uuid)))

        return ProjectionResult(data=CounterState(count=current_count), trace=trace)


class Increment(BaseModel):
    value: int = 1


def test_entity_facet_flow() -> None:
    con = ibis.duckdb.connect(":memory:")
    ledger = Ledger(con, "events")
    ledger.ensure()

    ledger.insert(Increment(value=1))
    ledger.insert(Increment(value=2))

    with Session(ledger) as sess:
        # 2. Read the entity (Facet)
        counter = CounterEntity(identity="my_counter")
        result = sess.Read(counter)
        assert result.data.count == 3
        assert len(result.trace) == 2

        # 3. Save a snapshot (This is a result of the session)
        counter.save(sess, result)

    # 4. Verify snapshot exists in ledger
    df = ledger.t.execute()
    # Should have 2 increments + 1 snapshot (CounterState)
    assert len(df) == 3
    snap_row = df[df["type"] == "CounterState"].iloc[0]
    assert snap_row["attributes"]["entity_identity"] == "my_counter"
    assert snap_row["payload"]["count"] == 3

    # 5. Commit more data BEFORE second session
    ledger.insert(Increment(value=10))

    # 6. Start a new session and verify reconstruction from snapshot
    with Session(ledger) as sess2:
        # Read again - should use the snapshot + new increment
        counter2 = CounterEntity(identity="my_counter")
        result2 = sess2.Read(counter2)

        assert result2.data.count == 13
        # Trace should be [SnapshotID, NewIncrementID]
        assert len(result2.trace) == 2


if __name__ == "__main__":
    pytest.main([__file__])
