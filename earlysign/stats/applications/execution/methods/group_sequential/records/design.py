"""Group sequential design records."""

from typing import Type

from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.stats.applications.design.group_sequential.initial_design.schema import (
    DesignPayloadModel,
)


class GroupSequentialDesignRecord(LedgerRecord, QueryMixin):
    """Persisted design snapshots for group sequential testing."""

    schema: dict[str, object] = {}

    @property
    def schema_pydantic_model(self) -> Type[DesignPayloadModel]:
        """Use the canonical design payload model for validation."""

        return DesignPayloadModel
