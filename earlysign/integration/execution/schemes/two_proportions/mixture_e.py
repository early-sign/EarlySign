"""Mixture e-process operators for two-proportion experiments."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.methods.group_sequential.schemes.two_proportions.binomial_arms import (
    BinomialArmSnapshot,
)
from earlysign.stats.schemes.two_proportions.statistic.mixture_e import (
    mixture_e_two_proportions,
    mixture_e_two_proportions_skew,
)

BetaPrior = Tuple[float, float]


class EProcessRecord(LedgerRecord, QueryMixin):
    """E-process snapshots for anytime-valid (safe) testing."""

    schema = {
        "E": (float | None, None),
        "logE": (float | None, None),
        "look": (int | None, None),
        "priors": (dict | None, None),
    }


class MixtureEProcessTwoProportions(LedgerOp):
    """Insert symmetric-alt mixture e-process row for two-proportions."""

    out_id: str
    control: BinomialArmSnapshot
    variant: BinomialArmSnapshot

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        control: BinomialArmSnapshot,
        variant: BinomialArmSnapshot,
        out_id: str,
        prior_null: BetaPrior = (0.5, 0.5),
        prior_alt: BetaPrior = (0.5, 0.5),
    ):
        super().__init__(
            ledger,
            control=control,
            variant=variant,
            out_id=out_id,
            prior_null=prior_null,
            prior_alt=prior_alt,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.eproc
        prior_null = tuple(getattr(self, "prior_null"))
        prior_alt = tuple(getattr(self, "prior_alt"))

        control_payload = self.control.latest_payload()
        variant_payload = self.variant.latest_payload()
        nA = int(control_payload["trial"])
        mA = int(control_payload["success"])
        nB = int(variant_payload["trial"])
        mB = int(variant_payload["success"])
        e_value = mixture_e_two_proportions(
            nA, mA, nB, mB, prior_null=prior_null, prior_alt=prior_alt
        )
        payload: Dict[str, Any] = {
            "E": float(e_value),
            "logE": float(math.log(max(e_value, 1e-300))),
            "priors": {"null": list(prior_null), "alt": list(prior_alt)},
        }
        out.insert(payload)


class MixtureEProcessTwoProportionsSkew(LedgerOp):
    """Insert skewed-alt mixture e-process row for two-proportions."""

    out_id: str
    control: BinomialArmSnapshot
    variant: BinomialArmSnapshot

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        control: BinomialArmSnapshot,
        variant: BinomialArmSnapshot,
        out_id: str,
        prior_null: BetaPrior = (0.5, 0.5),
        prior_alt_A: BetaPrior = (0.5, 0.5),
        prior_alt_B: BetaPrior = (1.0, 0.5),
    ):
        super().__init__(
            ledger,
            control=control,
            variant=variant,
            out_id=out_id,
            prior_null=prior_null,
            prior_alt_A=prior_alt_A,
            prior_alt_B=prior_alt_B,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.eproc
        prior_null = tuple(getattr(self, "prior_null"))
        prior_alt_A = tuple(getattr(self, "prior_alt_A"))
        prior_alt_B = tuple(getattr(self, "prior_alt_B"))

        control_payload = self.control.latest_payload()
        variant_payload = self.variant.latest_payload()
        nA = int(control_payload["trial"])
        mA = int(control_payload["success"])
        nB = int(variant_payload["trial"])
        mB = int(variant_payload["success"])
        e_value = mixture_e_two_proportions_skew(
            nA,
            mA,
            nB,
            mB,
            prior_null=prior_null,
            prior_alt_A=prior_alt_A,
            prior_alt_B=prior_alt_B,
        )
        payload: Dict[str, Any] = {
            "E": float(e_value),
            "logE": float(math.log(max(e_value, 1e-300))),
            "priors": {
                "null": list(prior_null),
                "alt_A": list(prior_alt_A),
                "alt_B": list(prior_alt_B),
            },
        }
        out.insert(payload)
