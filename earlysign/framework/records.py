from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol
from ibis.expr.types import Table as TableExpr
from earlysign.core.ledger import Ledger


class HasTable(Protocol):
    @property
    def t(self) -> TableExpr: ...


class QueryMixin:
    """
    Helpers for ledger-backed records.

    Assumes the concrete class provides:
      - property `t: TableExpr`.
    """

    def latest(self: HasTable) -> TableExpr:
        t = self.t
        return t.order_by(t.ts.desc()).limit(1)

    def order_by_ts(self: HasTable, ascending: bool = True) -> TableExpr:
        t = self.t
        keys = (t.ts.asc(), t.uuid.asc()) if ascending else (t.ts.desc(), t.uuid.desc())
        return t.order_by(*keys)

    def since(self: HasTable, ts: Any) -> TableExpr:
        t = self.t
        return t.filter(t.ts >= ts)

    def between(
        self: HasTable, start: Any, end: Any, *, include_end: bool = False
    ) -> TableExpr:
        t = self.t
        expr = t.filter(t.ts >= start)
        return expr.filter(t.ts <= end) if include_end else expr.filter(t.ts < end)


@dataclass
class LedgerRecord:
    """
    Typed view over ledger rows for a given payload_type and record_id.

    - `id` is REQUIRED: every persisted row is tagged with labels["record_id"] = id
    - `payload_type` is set by subclasses (no @dataclass on subclasses to avoid init clashes)
    - `ledger` is attached via .attach(ledger)
    """

    id: str
    payload_type: str = field(init=False, default="")
    ledger: Optional[Ledger] = field(default=None, init=False, repr=False)

    def attach(self, ledger: Ledger) -> "LedgerRecord":
        self.ledger = ledger
        return self

    @property
    def t(self) -> TableExpr:
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")
        return self.ledger.t.filter(
            self.ledger.t.payload_type == self.payload_type
        ).filter(self.ledger.t.labels["record_id"].str == str(self.id))

    def insert(
        self,
        payload: Mapping[str, Any],
        *,
        labels: Mapping[str, Any] = {},
    ) -> None:
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")
        self.ledger.insert(
            payload_type=self.payload_type,
            payload=payload,
            labels={**labels, "record_id": self.id},
        )
