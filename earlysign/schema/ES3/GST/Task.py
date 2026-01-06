from typing import List, Literal, Optional

from pydantic import BaseModel

from earlysign.schema.ES3.Base import TaskSpec as BaseTaskSpec
from earlysign.schema.ES3.GST.Hypothesis import HypothesisSpec


class EfficacyRequirement(BaseModel):
    alpha: float


class FutilityRequirement(BaseModel):
    power: float
    binding: bool = False


class TaskSpec(BaseTaskSpec):
    kind: Literal["group_sequential"] = "group_sequential"
    arms: List[str]
    response_type: Literal["binary", "continuous", "time_to_event"]
    hypotheses: HypothesisSpec

    efficacy: Optional[EfficacyRequirement] = None
    futility: Optional[FutilityRequirement] = None
