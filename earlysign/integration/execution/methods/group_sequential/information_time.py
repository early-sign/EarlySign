"""Information time record and operators for group sequential workflows."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.integration.execution.schemes.two_proportions.binomial_arms import (
    BinomialArmSnapshot,
)
from earlysign.stats.methods.group_sequential.info_time import (
    info_time_from_fisher,
    info_time_from_ratio,
    info_time_from_sample_size,
    info_time_from_sd,
    info_time_from_variance,
)


class InformationTimeRecord(LedgerRecord, QueryMixin):
    """
    Information time snapshots (scheme-agnostic).

    Stores information time t in [0, 1].

    Payload example:
      {"info_time": 0.5}
    """

    schema = {
        "info_time": float,
    }


class InformationTime(LedgerOp):
    """
    Insert information-time record from sample counts.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    out_id : str
        ID of InformationTimeRecord to create.
    control : BinomialArmSnapshot
        Record containing cumulative counts for the control arm.
    variants : Sequence[BinomialArmSnapshot]
        Record(s) containing cumulative counts for the comparison arm(s).
    planned_max_n : int
        Maximum total sample size.
    """

    out_id: str
    control: BinomialArmSnapshot
    variants: tuple[BinomialArmSnapshot, ...]
    planned_max_n: int

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        control: BinomialArmSnapshot,
        variants: BinomialArmSnapshot | Sequence[BinomialArmSnapshot],
        planned_max_n: int,
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            control=control,
            variants=tuple(variants) if isinstance(variants, Sequence) else (variants,),
            planned_max_n=planned_max_n,
        )

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        if not self.variants:
            raise ValueError("At least one variant record is required.")

        control_payload = self.control.latest_payload()
        variant_payloads = [rec.latest_payload() for rec in self.variants]

        trial_control = int(control_payload["trial"])
        trial_variants = sum(int(payload["trial"]) for payload in variant_payloads)
        n_total = trial_control + trial_variants

        t = info_time_from_sample_size(n_current=n_total, n_max=int(self.planned_max_n))
        out.insert({"info_time": float(t)})


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
