from dataclasses import dataclass
from earlysign.framework.record import LedgerRecord
from earlysign.framework.traits import QueryMixin


@dataclass(frozen=True, kw_only=True)
class InformationTimeRecord(QueryMixin, LedgerRecord):
    payload_type: str = "InfoTime"


@dataclass(frozen=True, kw_only=True)
class GSTBoundaryRecord(QueryMixin, LedgerRecord):
    payload_type: str = "GSTBoundary"


@dataclass(frozen=True, kw_only=True)
class DecisionSignalRecord(QueryMixin, LedgerRecord):
    payload_type: str = "Decision"
