from typing import Any, Mapping, Optional
from earlysign.framework.records import LedgerRecord, QueryMixin


class InformationTimeRecord(LedgerRecord, QueryMixin):
    """Stores the information time (e.g., n / Nmax or planned fraction)."""

    payload_type = "InfoTime"

    def insert(self, info_time: float, labels: Optional[Mapping[str, Any]] = None) -> None:  # type: ignore[override]
        payload = {"info_time": float(info_time)}
        super().insert(payload, labels=labels or {})


class GSTBoundaryRecord(LedgerRecord, QueryMixin):
    """Stores the current upper/lower GST boundaries given information time."""

    payload_type = "GSTBoundary"

    def insert(  # type: ignore[override]
        self,
        upper: float,
        lower: float,
        alpha: float,
        style: str,
        labels: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = {
            "boundary_upper": float(upper),
            "boundary_lower": float(lower),
            "alpha": float(alpha),
            "style": str(style),
        }
        super().insert(payload, labels=labels or {})


class DecisionSignalRecord(LedgerRecord, QueryMixin):
    """Stores the gating decision and the evidence values used."""

    payload_type = "DecisionSignal"

    def insert(  # type: ignore[override]
        self,
        signal: str,
        wald_z: float,
        info_time: float,
        labels: Mapping[str, Any] = {},
    ) -> None:
        payload = {
            "signal": str(signal),
            "wald_z": float(wald_z),
            "info_time": float(info_time),
        }
        super().insert(payload, labels=labels)
