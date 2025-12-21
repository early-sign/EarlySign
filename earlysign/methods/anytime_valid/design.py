"""Pydantic schema for anytime-valid (safe) testing designs."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from earlysign.methods.anytime_valid.records import SafeDesignRecord


class SafeDesignModel(BaseModel):
    """Design payload for safe testing."""

    alpha: float = Field(..., gt=0.0, lt=1.0)
    futility: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

    def to_payload(self) -> Dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        return dict(payload)


def insert_safe_design(record: SafeDesignRecord, payload: Dict[str, Any]) -> None:
    """Validate and insert a safe design payload into the provided record."""

    model = SafeDesignModel.model_validate(payload)
    record.insert(model.to_payload())
