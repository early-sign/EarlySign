"""Boundary record and operator for group sequential testing."""

from dataclasses import dataclass
from typing import Dict, Optional

from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.integration.execution.methods.group_sequential.information_time import (
    InformationTimeRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.design import (
    DesignPayloadModel,
    GroupSequentialDesignRecord,
)
from earlysign.stats.methods.group_sequential import boundary


class GroupSequentialBoundaryRecord(LedgerRecord, QueryMixin):
    """
    Nominal boundaries resolved from a GS design.

    Stores efficacy and futility boundaries resolved at a given
    information time, plus design metadata.
    """

    schema = {
        "upper": float,
        "lower": float,
        "efficacy": (dict | None, None),
        "futility": (dict | None, None),
        "scale": str,
        "statistic_type": (str | None, None),
        "statistic_scale": (str | None, None),
        "alpha": (float | None, None),
        "tails": (int | None, None),
        "info_time": (float | None, None),
        "look": (int | None, None),
    }


class BoundaryFromDesign(LedgerOp):
    """
    Read latest Design & InfoTime, compute and insert boundary record.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    design : GroupSequentialDesignRecord (attached)
        Design record to read from.
    info : InformationTimeRecord (attached)
        Information time record to read from.
    out_id : str
        ID of boundary record to create.
    look : int, optional
        Look number (1-indexed), required for significance_level style.
    """

    design: GroupSequentialDesignRecord
    info: InformationTimeRecord
    out_id: str
    look: Optional[int]

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        boundary: GroupSequentialBoundaryRecord

    outputs: Outputs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {
            "boundary": GroupSequentialBoundaryRecord(
                name=self.out_id, ledger=self.ledger
            )
        }

    def run(self) -> None:
        design_rec = self.design
        info_rec = self.info
        out = self.outputs.boundary
        look = getattr(self, "look", None)

        # Read latest design
        try:
            design_payload = design_rec.latest_payload()
        except LookupError:
            return
        design_model = DesignPayloadModel.model_validate(design_payload)
        boundary_spec = design_model.boundary_spec()

        # Read latest info time
        try:
            info_payload = info_rec.latest_payload()
        except LookupError:
            return
        t = float(info_payload["info_time"])

        # Compute boundaries using the canonical BoundaryCalculator API
        calc = boundary.BoundaryCalculator(spec=boundary_spec, process=None)
        up, lo, scale = calc.compute_boundary(info_time=t, look=look)

        # Compose payload
        payload = {
            "upper": float(up),
            "lower": float(lo),
            "efficacy": {"upper": float(up)},
            "futility": {
                "lower": float(lo),
                "binding": design_model.futility.is_binding,
                "mode": str(design_model.futility.mode.value),
            },
            "scale": str(scale),
            "alpha": float(design_model.alpha),
            "tails": int(design_model.hypothesis.tails),
            "info_time": float(t),
        }
        payload["statistic_type"] = design_model.statistic.kind.value
        payload["statistic_scale"] = design_model.statistic.scale.value
        if look is not None:
            payload["look"] = int(look)

        out.insert(payload)
