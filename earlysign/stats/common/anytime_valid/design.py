"""
Design operators for Anytime-Valid (Safe) testing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from earlysign.framework.operator import LedgerOperator, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.anytime_valid.records import SafeDesignRecord


class SafeDesign(LedgerOperator):
    """
    Persist a safe-testing design.

    __init__ parameters
    -------------------
    out_id   : str
    alpha    : float in (0,1)
    futility : Optional[dict] = None

    Examples
    --------
    out_id: "safe:design"
    alpha: 0.05
    futility: {"mode": "fixed", "tau": 0.1}
    """

    out_id: str
    alpha: float
    futility: Optional[Dict[str, Any]]

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        design: SafeDesignRecord

    outputs: Outputs

    def derived_records(self) -> dict[str, LedgerRecord]:
        return {"design": SafeDesignRecord(name=self.out_id)}

    def run(self) -> None:
        payload: Dict[str, Any] = {"alpha": float(self.alpha)}
        if getattr(self, "futility", None) is not None:
            payload["futility"] = dict(self.futility)  # type: ignore[arg-type]
        self.outputs.design.insert(payload)
