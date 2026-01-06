from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from earlysign.schema.ES3.GST.ReferenceModel import (
    BinaryModel,
    ContinuousModel,
    SurvivalModel,
)


class BoundarySpec(BaseModel):
    reference_model: Union[BinaryModel, ContinuousModel, SurvivalModel] = Field(
        ..., discriminator="kind"
    )
    boundary_scale: Literal["z_score", "p_value", "score_statistic"] = "z_score"
    binding: bool = True
    kind: str


class SpendingFunctionSpec(BaseModel):
    type: Literal[
        "obrien_fleming",
        "pocock",
        "kim_demets",
        "lan_demets",
        "power_family",
        "hwang_shih_decani",
        "custom",
    ]
    params: Optional[Dict[str, float]] = None


class SpendingBoundary(BoundarySpec):
    kind: Literal["spending"] = "spending"
    spending_function: SpendingFunctionSpec


class FixedBoundary(BoundarySpec):
    kind: Literal["fixed"] = "fixed"
    value: float


class CustomBoundary(BoundarySpec):
    kind: Literal["custom"] = "custom"
    values: List[float]


class WangTsiatisBoundary(BoundarySpec):
    kind: Literal["wang_tsiatis"] = "wang_tsiatis"
    delta: float


class ScheduleSpec(BaseModel):
    unit: Literal["sample_size", "information_fraction", "calendar_time", "events"]
    n_looks: Optional[int] = None
    interim_points: Optional[List[float]] = None


class StoppingRule(BaseModel):
    schedule: ScheduleSpec
    boundary: Union[
        SpendingBoundary, FixedBoundary, CustomBoundary, WangTsiatisBoundary
    ] = Field(..., discriminator="kind")
