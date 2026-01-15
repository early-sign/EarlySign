from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterator

from earlysign.core.ledger import Ledger
from earlysign.v0.framework.records import LedgerRecord


class LedgerOpOutputs(ABCMapping[str, LedgerRecord]):
    """Base class for typed operator outputs.

    Provides both attribute access (outputs.snapshot) and dict-like access (outputs["snapshot"]).
    Designed to be used with @dataclass(frozen=True).

    Usage:
        @dataclass(frozen=True)
        class Outputs(LedgerOpOutputs):
            snapshot: MySnapshotRecord

    This enables:
    - Type-safe attribute access: self.outputs.snapshot
    - Dict-like access: self.outputs["snapshot"]
    - Mapping compatibility for generic operations
    """

    def __getitem__(self, key: str) -> LedgerRecord:
        """Dict-like access to fields."""
        try:
            value = getattr(self, key)
            if not isinstance(value, LedgerRecord):
                raise TypeError(
                    f"Field '{key}' is not a LedgerRecord, got {type(value)}"
                )
            return value
        except AttributeError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(f.name for f in fields(self))  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Number of fields."""
        return len(fields(self))  # type: ignore[arg-type]


@dataclass
class LedgerOp:
    """
    __init__(ledger, **inputs):
      - ledger: Ledger
      - inputs: attached LedgerRecord(s) or required dependencies
    Subclasses must override:
      - build_outputs() -> Dict[str, LedgerRecord]  # ledger-attached outputs
      - run() -> dict                               # no args; use self.outputs[...] to insert
    Note:
      - Store outputs only in self.outputs (do not add attributes)
    """

    outputs: Any  # subclasses override with typed Outputs

    def __init__(self, ledger: Ledger, **inputs: Any):
        self.ledger = ledger
        for k, v in inputs.items():
            setattr(self, k, v)

        built = self.build_outputs()
        outs: Dict[str, LedgerRecord] = {}
        for name, rec in built.items():
            if rec.ledger is None:
                raise RuntimeError(
                    f"Output record '{name}' must be attached to a ledger before return."
                )
            outs[name] = rec

        if hasattr(self.__class__, "Outputs"):
            outputs_cls = getattr(self.__class__, "Outputs")
            self.outputs = outputs_cls(**outs)
        else:
            self.outputs = outs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        """Return ledger-attached output records.

        Subclasses should override this and specify their own return type.
        The return value must be a mapping of output names to LedgerRecord instances
        that have already been attached to a ledger (typically `self.ledger`).
        """
        return {}

    def run(self) -> Any:
        raise NotImplementedError
