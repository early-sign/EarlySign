from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypeAliasType

from earlysign.builtin.group_sequential.schema.enums import Unit


class InformationTimerBase(BaseModel):
    kind: str


class EventCountTimer(InformationTimerBase):
    kind: Literal["event_count"] = "event_count"
    max_events: int


class FisherInformationTimer(InformationTimerBase):
    kind: Literal["fisher_information"] = "fisher_information"
    max_information: float


class SampleSizeTimer(InformationTimerBase):
    kind: Literal["sample_size"] = "sample_size"
    unit: Unit
    max_sample_size: dict[str, int]


InformationTimer = TypeAliasType(
    "InformationTimer",
    Annotated[
        SampleSizeTimer | FisherInformationTimer | EventCountTimer,
        Field(..., description="Discriminated Union for Timer."),
    ],
)


class ScheduleBase(BaseModel):
    kind: str


class EquidistantSchedule(ScheduleBase):
    kind: Literal["equidistant"] = "equidistant"
    n_looks: int


class FixedSchedule(ScheduleBase):
    kind: Literal["fixed"] = "fixed"
    analyses: list[float]


ScheduleSpec = TypeAliasType(
    "ScheduleSpec",
    Annotated[
        FixedSchedule | EquidistantSchedule,
        Field(..., description="Discriminated Union for Schedule."),
    ],
)
