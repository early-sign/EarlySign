"""
Post-hoc inspection utilities for Group Sequential Testing.
"""

from typing import Optional

import ibis
from ibis import _

from earlysign.v1.framework.projector import ProjectionResult, Projector


class PosthocZTrajectoryProjector(Projector[ibis.Expr]):
    """
    Projector that reconstructs the batch-wise Z-trajectory from the ledger.
    Returns an Ibis expression builder.
    """

    def __init__(
        self,
        arm_data_type: str = "BinomialArmData",
        control_arm: Optional[str] = None,
        treatment_arm: Optional[str] = None,
    ):
        self.arm_data_type = arm_data_type
        self.control_arm = control_arm
        self.treatment_arm = treatment_arm

    def project(self, table: ibis.Expr) -> ProjectionResult[ibis.Expr]:
        # 1. Filter and prepare ArmData
        arm_data = table.filter(_.type == self.arm_data_type)

        # 1a. Identify Arms from Protocol if not provided
        control = self.control_arm
        treatment = self.treatment_arm

        if control is None or treatment is None:
            # Look for the latest Protocol in the ledger
            protocol_df = (
                table.filter(table.type.like("%Protocol"))
                .order_by(ibis.desc("timestamp"))
                .limit(1)
                .execute()
            )
            if not protocol_df.empty:
                import json

                payload = protocol_df.iloc[0]["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)

                # Identify arms from the new polymorphic 'arms' structure
                arms_cfg = payload.get("task", {}).get("arms", {})
                if arms_cfg.get("kind") == "two_arm":
                    if control is None:
                        control = arms_cfg.get("control_arm_name")
                    if treatment is None:
                        treatment = arms_cfg.get("treatment_arm_name")

        if control is None or treatment is None:
            raise ValueError(
                "Could not identify control and treatment arms from Protocol. "
                "Explicit arm roles (TwoArmComparison) are required for Z-trajectory reconstruction."
            )

        # 2. Extract fields from payload
        def _get_val(col: str) -> ibis.Expr:
            return _.payload[col].cast("string").re_replace('^"|"$', "")

        from earlysign.core.util.json_ops import extract_json_scalar

        # We use extract_json_scalar here to avoid BigQuery's strict JSON-to-INT64 cast
        # which can fail on stringified numbers or non-conforming rows in unions.
        arm_data = arm_data.mutate(
            arm=_get_val("arm"),
            n_val=extract_json_scalar(arm_data.payload, "total", "int"),
            s_val=extract_json_scalar(arm_data.payload, "success", "int"),
        )

        # 3. Stable ordering and cumulative window
        arm_data = arm_data.mutate(
            idx=ibis.row_number().over(ibis.window(order_by=_.timestamp))
        )
        win = ibis.window(order_by=_.idx, preceding=None, following=0)

        # 4. Calculate cumulative stats and Z-score
        trajectory = (
            arm_data.mutate(
                cnc=(_.arm == control).ifelse(_.n_val, 0).sum().over(win),
                csc=(_.arm == control).ifelse(_.s_val, 0).sum().over(win),
                cnt=(_.arm == treatment).ifelse(_.n_val, 0).sum().over(win),
                cst=(_.arm == treatment).ifelse(_.s_val, 0).sum().over(win),
            )
            .filter((_.cnc > 0) & (_.cnt > 0))
            .mutate(
                pc=_.csc / _.cnc,
                pt=_.cst / _.cnt,
                pp=(_.csc + _.cst) / (_.cnc + _.cnt),
            )
            .mutate(se=(_.pp * (1 - _.pp) * (1 / _.cnc + 1 / _.cnt)).sqrt())
            .mutate(z=(_.se > 0).ifelse((_.pt - _.pc) / _.se, 0.0))
            .select(n=_.cnc + _.cnt, z=_.z)
        )

        # For Projector interface, we return the expression as data
        # Trace is empty as this is a collective reconstruction
        return ProjectionResult(data=trajectory, trace=[])
