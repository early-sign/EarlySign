from dataclasses import dataclass, field
from typing import Literal, Optional

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.types import TestType


@dataclass
class TimeToEventEffect:
    hazard_ratio: float = 0.6


@dataclass
class TimeToEventSampleSize:
    total_events: int = 171
    total_sample_size: int = 296
    time_unit: Literal["days", "weeks", "months", "years"] = "months"
    accrual_duration: Optional[float] = None
    follow_up_duration: Optional[float] = None


@dataclass
class TimeToEventDesignSpec(DesignSpec):
    effect: TimeToEventEffect = field(default_factory=TimeToEventEffect)
    sample_size: TimeToEventSampleSize = field(default_factory=TimeToEventSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TIME_TO_EVENT
