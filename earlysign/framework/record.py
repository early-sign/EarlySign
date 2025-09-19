from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Dict, Any, Self
from ibis.expr.types import Table as TableExpr
from earlysign.core.ledger import Ledger


@dataclass(frozen=True)
class LedgerRecord:
    payload_type: str
    id: str
    ledger: Optional[Ledger] = field(default=None, repr=False, compare=False)

    def attach(self, ledger: Ledger) -> Self:
        return replace(self, ledger=ledger)

    @property
    def t(self) -> TableExpr:
        if self.ledger is None:
            raise RuntimeError("LedgerRecord is not attached; call attach(...)")
        base = self.ledger.t
        return base.filter(
            base.payload_type.cast("string") == self.payload_type
        ).filter(base.labels["id"].cast("string") == self.id)

    def insert(
        self, payload: Mapping[str, Any], labels: Optional[Mapping[str, Any]] = None
    ) -> str:
        if self.ledger is None:
            raise RuntimeError("LedgerRecord is not attached; call attach(...)")
        labs: Dict[str, Any] = {"id": self.id}
        if labels:
            labs.update(labels)
        return self.ledger.insert(
            payload_type=self.payload_type, payload=dict(payload), labels=labs
        )
