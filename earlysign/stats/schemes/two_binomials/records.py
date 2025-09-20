from typing import Any, Mapping, Optional, Dict
from earlysign.framework.records import LedgerRecord, QueryMixin


class BinomialCountsRecord(LedgerRecord, QueryMixin):
    """Holds cumulative counts for two-binomial A/B snapshots."""

    payload_type = "BinomCounts"

    def insert(  # type: ignore[override]
        self,
        nA: int,
        mA: int,
        nB: int,
        mB: int,
        labels: Mapping[str, Any] = {},
    ) -> None:
        payload = {"nA": int(nA), "mA": int(mA), "nB": int(nB), "mB": int(mB)}
        super().insert(payload, labels=labels)


class WaldZStatisticRecord(LedgerRecord, QueryMixin):
    """Stores the computed Wald Z and components."""

    payload_type = "WaldZ"

    def insert(  # type: ignore[override]
        self,
        wald_z: float,
        pA: float,
        pB: float,
        se: float,
        labels: Mapping[str, Any] = {},
    ) -> None:
        payload = {
            "wald_z": float(wald_z),
            "pA": float(pA),
            "pB": float(pB),
            "se": float(se),
        }
        super().insert(payload, labels=labels)
