"""Information time operators for generic group sequential workflows."""

from dataclasses import dataclass
from typing import Dict, Optional

from earlysign.integration.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.essentials.methods.group_sequential.info_time import (
    info_time_from_fisher,
    info_time_from_ratio,
    info_time_from_sd,
    info_time_from_variance,
)


class InformationTimeFromRatio(LedgerOp):
    """Insert t = info_now / info_max."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        info_now: float,
        info_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger, out_id=out_id, info_now=info_now, info_max=info_max, look=look
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_ratio(
            info_now=float(getattr(self, "info_now")),
            info_max=float(getattr(self, "info_max")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromVariance(LedgerOp):
    """Insert t = var_target / var_now."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        var_now: float,
        var_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger, out_id=out_id, var_now=var_now, var_target=var_target, look=look
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_variance(
            var_now=float(getattr(self, "var_now")),
            var_target=float(getattr(self, "var_target")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromSD(LedgerOp):
    """Insert t = (sd_target^2) / (sd_now^2)."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        sd_now: float,
        sd_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger, out_id=out_id, sd_now=sd_now, sd_target=sd_target, look=look
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_sd(
            sd_now=float(getattr(self, "sd_now")),
            sd_target=float(getattr(self, "sd_target")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromFisher(LedgerOp):
    """Insert t = fisher_now / fisher_max."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        fisher_now: float,
        fisher_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            fisher_now=fisher_now,
            fisher_max=fisher_max,
            look=look,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_fisher(
            fisher_now=float(getattr(self, "fisher_now")),
            fisher_max=float(getattr(self, "fisher_max")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)
