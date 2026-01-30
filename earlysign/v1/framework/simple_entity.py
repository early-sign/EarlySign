from typing import Optional, Type, TypeVar

import ibis

from earlysign.v1.framework.base_entity import BaseEntity
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId

T = TypeVar("T")


class SimpleEntity(BaseEntity[Optional[T]]):
    """
    A lightweight projection for 'the latest value' of a specific identity.

    Unlike Entity, SimpleEntity does not support incremental computation (fold).
    It simply looks for the latest record of the given schema type matching
    the identity in its labels.
    """

    def __init__(self, data_type: Type[T], identity: str):
        super().__init__(identity)
        self.data_type = data_type

    def project(self, table: ibis.Expr) -> ProjectionResult[Optional[T]]:
        """
        Find the latest record for this identity.
        """
        schema_name = self.data_type.__name__
        matched = table.filter(
            (table.type == schema_name)
            & (table.attributes["entity_identity"].str == self.identity)
        )

        latest = matched.order_by(ibis.desc("timestamp")).limit(1).execute()

        if latest.empty:
            # For SimpleEntity, we might want an initial value or None.
            # Here we follow the Projector interface but it might return None data.
            return ProjectionResult(data=None, trace=[])

        row = latest.iloc[0]
        # Payload is the data
        data_raw = row["payload"]
        data_inst = (
            self.data_type(**data_raw) if isinstance(data_raw, dict) else data_raw
        )

        return ProjectionResult(
            data=data_inst,
            trace=[TraceId(str(row["uuid"]))],
        )
