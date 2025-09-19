from dataclasses import dataclass
from typing import Dict, Any
from earlysign.framework.record import LedgerRecord
from earlysign.framework.scoped import ScopedLedger


@dataclass
class LedgerOperator:
    """
    __init__(scoped, **inputs):
      - scoped: ScopedLedger
      - inputs: attached LedgerRecord(s) or必要な依存
    Subclass must override:
      - derived_records() -> Dict[str, LedgerRecord]  # unattached outputs
      - run() -> dict                                 # 引数なし。self.outputs[...] を使って insert する
    Note:
      - 出力は self.outputs にのみ格納する（属性は生やさない）
    """

    def __init__(self, scoped: ScopedLedger, **inputs: Any):
        self.scoped = scoped
        for k, v in inputs.items():
            setattr(self, k, v)

        outs: Dict[str, LedgerRecord] = {}
        for name, rec in self.derived_records().items():
            outs[name] = rec.attach(scoped)
        self._outputs = outs

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {}

    @property
    def outputs(self) -> Dict[str, LedgerRecord]:
        return self._outputs

    def run(self) -> Any:
        raise NotImplementedError
