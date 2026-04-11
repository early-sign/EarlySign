from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from typing_extensions import TypeAliasType


class HypothesisParametersBase(BaseModel):
    kind: str


class EqualityHypothesis(HypothesisParametersBase):
    kind: Literal["equality"] = "equality"


class EquivalenceHypothesis(HypothesisParametersBase):
    kind: Literal["equivalence"] = "equivalence"
    lower_margin: float
    upper_margin: float


class NonInferiorityHypothesis(HypothesisParametersBase):
    kind: Literal["non_inferiority"] = "non_inferiority"
    non_inferiority_margin: float


class SuperiorityHypothesis(HypothesisParametersBase):
    kind: Literal["superiority"] = "superiority"
    superiority_margin: float


HypothesisParameters = TypeAliasType(
    "HypothesisParameters",
    EqualityHypothesis
    | SuperiorityHypothesis
    | NonInferiorityHypothesis
    | EquivalenceHypothesis,
)


class EffectSizeSpecBase(BaseModel):
    type: str


class BinaryEffectSize(EffectSizeSpecBase):
    type: Literal["binary"] = "binary"
    proportions: dict[str, float]


class ContinuousEffectSize(EffectSizeSpecBase):
    type: Literal["continuous"] = "continuous"
    means: dict[str, float]
    standard_deviation: float


class SurvivalEffectSize(EffectSizeSpecBase):
    type: Literal["time_to_event"] = "time_to_event"
    hazard_ratios: dict[str, float]
    median_survival_times: dict[str, float] | None = None
    event_rate: float | None = None


EffectSizeUnion = TypeAliasType(
    "EffectSizeUnion", BinaryEffectSize | ContinuousEffectSize | SurvivalEffectSize
)


class HypothesisSpec(BaseModel):
    h_null_description: str
    h_alt_description: str
    test_logic: HypothesisParameters
    target_effect: EffectSizeUnion


class EffectMeasureBase(BaseModel):
    kind: str
    value: float


class AbsoluteDifference(EffectMeasureBase):
    kind: Literal["absolute_difference"] = "absolute_difference"


class OddsRatio(EffectMeasureBase):
    kind: Literal["odds_ratio"] = "odds_ratio"


class RelativeImprovement(EffectMeasureBase):
    kind: Literal["relative_improvement"] = "relative_improvement"


class RelativeRisk(EffectMeasureBase):
    kind: Literal["relative_risk"] = "relative_risk"


EffectMeasure = TypeAliasType(
    "EffectMeasure", AbsoluteDifference | OddsRatio | RelativeRisk | RelativeImprovement
)
