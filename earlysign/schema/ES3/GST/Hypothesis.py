from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


class HypothesisParameters(BaseModel):
    kind: str


class EqualityHypothesis(HypothesisParameters):
    kind: Literal["equality"] = "equality"


class SuperiorityHypothesis(HypothesisParameters):
    kind: Literal["superiority"] = "superiority"
    superiority_margin: float = 0.0


class NonInferiorityHypothesis(HypothesisParameters):
    kind: Literal["non_inferiority"] = "non_inferiority"
    non_inferiority_margin: float


class EquivalenceHypothesis(HypothesisParameters):
    kind: Literal["equivalence"] = "equivalence"
    lower_margin: float
    upper_margin: float


class EffectSizeSpec(BaseModel):
    type: str


class BinaryEffectSize(EffectSizeSpec):
    type: Literal["binary"] = "binary"
    proportions: Dict[str, float]


class ContinuousEffectSize(EffectSizeSpec):
    type: Literal["continuous"] = "continuous"
    means: Dict[str, float]
    standard_deviation: float


class SurvivalEffectSize(EffectSizeSpec):
    type: Literal["time_to_event"] = "time_to_event"
    hazard_ratios: Dict[str, float]
    median_survival_times: Optional[Dict[str, float]] = None
    event_rate: Optional[float] = None


class HypothesisSpec(BaseModel):
    h_null: str
    h_alt: str
    test_logic: Union[
        EqualityHypothesis,
        SuperiorityHypothesis,
        NonInferiorityHypothesis,
        EquivalenceHypothesis,
    ] = Field(..., discriminator="kind")
    target_effect: Union[BinaryEffectSize, ContinuousEffectSize, SurvivalEffectSize] = (
        Field(..., discriminator="type")
    )
