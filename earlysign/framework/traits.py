from typing import Any
from ibis.expr.types import Table as TableExpr


class QueryMixin:
    """
    Lightweight read helpers for LedgerRecord:
      - latest(): TableExpr (no execute)
      - order_by_ts(ascending=True|False): TableExpr
      - since(ts), between(start, end, include_end=False): TableExpr

    NOTE: JSON/labels access is intentionally *not* wrapped.
          Use native Ibis API, e.g. rec.t.payload["x"], rec.t.labels["id"].
    """

    def latest(self) -> TableExpr:
        t = getattr(self, "t")
        return t.order_by(t.ts.desc(), t.uuid.desc()).limit(1)

    def order_by_ts(self, ascending: bool = True) -> TableExpr:
        t = getattr(self, "t")
        key = (t.ts.asc(), t.uuid.asc()) if ascending else (t.ts.desc(), t.uuid.desc())
        return t.order_by(*key)

    def since(self, ts: Any) -> TableExpr:
        t = getattr(self, "t")
        return t.filter(t.ts >= ts)

    def between(self, start: Any, end: Any, *, include_end: bool = False) -> TableExpr:
        t = getattr(self, "t")
        expr = t.filter(t.ts >= start)
        return expr.filter(t.ts <= end) if include_end else expr.filter(t.ts < end)
