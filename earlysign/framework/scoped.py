from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Dict
from ibis import Table as TableExpr

from earlysign.core.ledger import Ledger


@dataclass
class ScopedLedger(Ledger):
    _bound: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def bind(self, **labels: Dict[str, str]) -> "ScopedLedger":
        merged = dict(self._bound)
        merged.update(labels)
        out = ScopedLedger(connector=self.connector, table_name=self.table_name)
        out._bound = merged
        return out

    @property
    def t(self) -> TableExpr:
        base = super().t
        t = base
        for k, v in self._bound.items():
            t = t.filter(base.labels[k].cast("string") == str(v))
        return t

    def insert(
        self,
        *,
        payload_type: str,
        payload: Mapping[str, Any],
        labels: Mapping[str, Any] = {},
    ) -> Any:
        labs = dict(self._bound)
        labs.update(labels)
        return super().insert(payload_type=payload_type, payload=payload, labels=labs)
