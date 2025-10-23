from dataclasses import dataclass, field
from typing import Literal, Optional

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.types import TestType


@dataclass
class ProportionsEffect:
    p_control: float = 0.10
    p_treatment: Optional[float] = None
    effect_size: Optional[float] = 0.02
    effect_type: Literal["absolute", "relative", "odds_ratio"] = "absolute"

    def get_treatment_proportion(self) -> float:
        if self.p_treatment is not None:
            return self.p_treatment
        if self.effect_size is None:
            raise ValueError("effect_size must be specified when p_treatment is None")
        if self.effect_type == "absolute":
            return self.p_control + self.effect_size
        elif self.effect_type == "relative":
            return self.p_control * (1 + self.effect_size)
        elif self.effect_type == "odds_ratio":
            odds_control = self.p_control / (1 - self.p_control)
            odds_treatment = odds_control * self.effect_size
            return odds_treatment / (1 + odds_treatment)
        else:
            raise ValueError(f"Unknown effect_type: {self.effect_type}")


@dataclass
class ProportionsSampleSize:
    n_per_analysis: int = 500


@dataclass
class ProportionsDesignSpec(DesignSpec):
    effect: ProportionsEffect = field(default_factory=ProportionsEffect)
    sample_size: ProportionsSampleSize = field(default_factory=ProportionsSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TWO_PROPORTIONS
