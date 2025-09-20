from dataclasses import dataclass
from typing import Dict, Any, Mapping
from earlysign.core.ledger import Ledger
from earlysign.framework.records import LedgerRecord


@dataclass
class LedgerOperator:
    """
    __init__(scoped, **inputs):
      - scoped: Ledger
      - inputs: attached LedgerRecord(s) or required dependencies
    Subclass must override:
      - derived_records() -> Dict[str, LedgerRecord]  # unattached outputs
      - run() -> dict                                 # no args; use self.outputs[...] to insert
    Note:
      - Store outputs only in self.outputs (do not add attributes)
    """

    def __init__(self, scoped: Ledger, **inputs: Any):
        self.scoped = scoped
        for k, v in inputs.items():
            setattr(self, k, v)

        outs: Dict[str, LedgerRecord] = {}
        for name, rec in self.derived_records().items():
            outs[name] = rec.attach(scoped)
        self._outputs = outs

    def derived_records(self) -> Mapping[str, LedgerRecord]:
        return {}

    @property
    def outputs(self) -> Mapping[str, LedgerRecord]:
        return self._outputs

    def run(self) -> Any:
        raise NotImplementedError
