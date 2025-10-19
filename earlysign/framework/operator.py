from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterator

from earlysign.core.ledger import Ledger
from earlysign.framework.records import LedgerRecord


class LedgerOpOutputs(ABCMapping[str, LedgerRecord]):
    """Base class for typed operator outputs.

    Provides both attribute access (outputs.snapshot) and dict-like access (outputs["snapshot"]).
    Designed to be used with @dataclass(frozen=True).

    Usage:
        @dataclass(frozen=True)
        class Outputs(LedgerOpOutputs):
            snapshot: BinomialCountsSnapshotRecord

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

    # Type annotation - subclasses can override with specific type
    outputs: Any

    def __init__(self, scoped: Ledger, **inputs: Any):
        self.scoped = scoped
        for k, v in inputs.items():
            setattr(self, k, v)

        outs: Dict[str, LedgerRecord] = {}
        for name, rec in self.derived_records().items():
            outs[name] = rec.attach(scoped)

        # If subclass has Outputs class, create instance; otherwise use dict
        if hasattr(self.__class__, "Outputs"):
            OutputsClass = getattr(self.__class__, "Outputs")
            self.outputs = OutputsClass(**outs)
        else:
            self.outputs = outs

    def derived_records(self) -> Dict[str, LedgerRecord]:
        """Return unattached output records.

        Subclasses should override this and specify their own return type.
        The return value should be a dict-like mapping of output names to LedgerRecord instances.
        """
        return {}

    def run(self) -> Any:
        raise NotImplementedError
