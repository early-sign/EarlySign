from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from earlysign.builtin.group_sequential.schema.enums import (
    DecisionStatus,
    PromisingZoneStatus,
)
from earlysign.schema.ES3.base import Log


class Trigger(BaseModel):
    kind: str


class ManualTrigger(Trigger):
    kind: Literal["manual"] = "manual"
    reason: str


class ScheduleTrigger(Trigger):
    kind: Literal["schedule"] = "schedule"
    index: int
    value: float


class AdaptationSnapshot(BaseModel):
    """
    State captured during an adaptation event to preserve Type I error control.
    """

    z_t: float = Field(
        ..., description="Observed Z-statistic at the time of adaptation."
    )
    info_frac: float = Field(
        ..., description="Information fraction at the time of adaptation (t)."
    )
    original_max_sample_size: int = Field(
        ..., description="The maximum sample size before re-planning."
    )
    use_weighted_statistic: bool | None = Field(
        True,
        description="Whether to use a weighted test statistic (e.g., Cui-Hung-Wang) after adaptation to preserve Type I error.",
    )


class AdaptationLog(Log):
    look: int
    conditional_power: float
    promising_zone_status: PromisingZoneStatus | str
    promising_zone_recommendation: str
    original_sample_size: int
    recommended_sample_size: int | None = None


class Analysis(Log):
    trigger: Trigger | None = None
    look: int
    statistic: float
    info_frac: float
    decision: DecisionStatus | str
    efficacy_boundary: float | None = None
    futility_boundary: float | None = None


class Decision(Log):
    status: DecisionStatus | str
    message: str


class LookResult(BaseModel):
    look: int | None = None
    trigger: Trigger | None = None
    sample_n: int
    info_frac: float
    z_stat: float
    z_stats: dict[str, float] | None = None
    efficacy_boundary: float | None = None
    is_efficacy_crossed: bool
    futility_boundary: float | None = None
    is_futility_crossed: bool
    alpha_spent: float | None = None
    beta_spent: float | None = None
    status: DecisionStatus | str
