from dataclasses import dataclass
from earlysign.framework.record import LedgerRecord
from earlysign.framework.traits import QueryMixin


@dataclass(frozen=True, kw_only=True)
class BinomialCountsRecord(QueryMixin, LedgerRecord):
    payload_type: str = "BinomCounts"


@dataclass(frozen=True, kw_only=True)
class WaldZStatisticRecord(QueryMixin, LedgerRecord):
    payload_type: str = "WaldZ"
