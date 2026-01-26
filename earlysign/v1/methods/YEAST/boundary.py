from typing import Optional

import ibis
import numpy as np
from scipy import stats

from earlysign.schema.ES3.YEAST import Protocol
from earlysign.schema.ES3.YEAST.Log import Boundary as BoundarySchema
from earlysign.v1.framework.entity import Entity, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId


class Boundary(Entity[BoundarySchema]):
    """
    Entity representing the fixed testing boundary for YEAST.

    This Entity calculates the boundary value based on the Protocol configuration.
    Since the boundary is fixed after design, this Entity primarily serves as a
    lookup for the `value`.

    The `compute` method allows it to be projected from the event stream,
    looking for `Boundary` payload types (emitted UpdateBoundary events),
    though typically it's initialized once.
    """

    data_type = BoundarySchema

    @classmethod
    def calculate(cls, protocol: Protocol) -> float:
        """
        Calculates the YEAST boundary value.

        B = z_{alpha/2} * sqrt(N_max * V_N)

        Args:
            protocol: The YEAST Protocol containing method parameters.

        Returns:
            The calculated boundary value.
        """
        method = protocol.method
        alpha = method.significance_level
        n_max = method.expected_num_observations
        estimated_variance = method.estimated_variance

        # Two-sided critical value (using upper tail)
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # b^* = z_{1 - \alpha/2} \sqrt{N \hat{V}_N}
        boundary_value = z_crit * np.sqrt(n_max * estimated_variance)
        return float(boundary_value)

    def compute(
        self,
        snapshot: Optional[Snapshot[BoundarySchema]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[BoundarySchema]:
        """
        Projects the Boundary state from the event stream.
        Logic: Use the latest 'Boundary' payload if available, else retain snapshot.
        """
        # Filter for Boundary updates
        boundary_updates = delta_expr.filter(delta_expr.payload_type == "Boundary")

        # Get latest update
        latest_df = (
            boundary_updates.order_by(boundary_updates.ts.desc())
            .limit(1)
            .select("uuid", "payload")
            .execute()
        )

        if not latest_df.empty:
            # Found a new boundary update
            row = latest_df.iloc[0]
            uuid = str(row["uuid"])
            payload = row["payload"]

            # If payload is string (some backends), parse it.
            # Ibis execution usually returns dict for JSON if backend supports it.
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            new_data = BoundarySchema.model_validate(payload)

            # Trace lineage
            trace = [TraceId(uuid)]
            if snapshot and snapshot.uuid:
                trace.insert(0, TraceId(str(snapshot.uuid)))

            return ProjectionResult(data=new_data, trace=trace)

        if snapshot:
            # No update, keep existing
            return ProjectionResult(
                data=snapshot.data, trace=[TraceId(str(snapshot.uuid))]
            )

        # No snapshot and no update -> Default state (should ideally not happen if properly initialized)
        return ProjectionResult(data=BoundarySchema(value=0.0), trace=[])
