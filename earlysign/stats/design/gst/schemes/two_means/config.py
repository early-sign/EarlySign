from dataclasses import dataclass, field
from typing import Optional

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.types import TestType


@dataclass
class MeansEffect:
    mean_control: float = 10.0
    mean_treatment: Optional[float] = None
    effect_size: Optional[float] = 2.0
    std_dev: float = 5.0
    pooled_std: bool = True


@dataclass
class MeansSampleSize:
    n_per_analysis: int = 100


@dataclass
class MeansDesignSpec(DesignSpec):
    effect: MeansEffect = field(default_factory=MeansEffect)
    sample_size: MeansSampleSize = field(default_factory=MeansSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TWO_MEANS
