"""
Boundary operator for group sequential testing.

Reads Design and InfoTime records, computes boundaries using essentials functions.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.integration.design.group_sequential.initial_design.schema import (
    DesignPayloadModel,
)
from earlysign.integration.execution.methods.group_sequential.records.boundary import (
    GroupSequentialBoundaryRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.design import (
    GroupSequentialDesignRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.stats.essentials.methods.group_sequential import boundary


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

    Examples
    --------
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.integration.execution.methods.group_sequential.operators.boundary import (
    ...     BoundaryFromDesign
    ... )
    >>> from earlysign.integration.execution.methods.group_sequential.records.design import (
    ...     GroupSequentialDesignRecord
    ... )
    >>> from earlysign.integration.execution.methods.group_sequential.records.info import (
    ...     InformationTimeRecord
    ... )
    >>> import ibis
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>>
    >>> # Write design
    >>> design_rec = GroupSequentialDesignRecord(name="design1").attach(ledger)
    >>> design_rec.insert(
    ...     {
    ...         "alpha": 0.05,
    ...         "hypothesis": {"structure": "two_sided_symmetric"},
    ...         "statistic": {"kind": "wald_z", "scale": "z"},
    ...         "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...         "futility": {"mode": "none", "binding_mode": "non_binding"},
    ...         "planned_max_n": 1000,
    ...         "planned_info_times": [0.5, 1.0],
    ...     }
    ... )
    >>>
    >>> # Write info time
    >>> info_rec = InformationTimeRecord(name="info1").attach(ledger)
    >>> info_rec.insert({"info_time": 0.5})
    >>>
    >>> # Compute boundary
    >>> boundary_op = BoundaryFromDesign(
    ...     ledger,
    ...     design=GroupSequentialDesignRecord(name="design1").attach(ledger),
    ...     info=info_rec,
    ...     out_id="bound1",
    ...     look=2
    ... )
    >>> boundary_op.run()
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
            # Flat fields for easy consumption
            "upper": float(up),
            "lower": float(lo),
            # Structured fields for richer consumers
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
