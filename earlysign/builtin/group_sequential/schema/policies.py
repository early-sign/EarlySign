from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from earlysign.builtin.group_sequential.schema.statistics import TestStatisticSpec
from earlysign.builtin.group_sequential.schema.strategies import DecisionStrategy
from earlysign.builtin.group_sequential.schema.timers import (
    InformationTimer,
    ScheduleSpec,
)


class DueLookTrigger(BaseModel):
    kind: Literal["due_look"] = "due_look"
    tolerance: float | None = 0


class TriggerStrategySpec(DueLookTrigger):
    """Discriminated Union for Trigger Strategies."""


class StoppingPolicySpec(BaseModel):
    statistic: TestStatisticSpec
    strategy: DecisionStrategy
    timer: InformationTimer
    schedule: ScheduleSpec
    trigger_strategy: TriggerStrategySpec | None = Field(
        default_factory=lambda: TriggerStrategySpec.model_validate(
            {"kind": "due_look", "tolerance": 0}
        )
    )
