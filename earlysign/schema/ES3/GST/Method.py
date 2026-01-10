from typing import List, Literal, Optional

from pydantic import BaseModel

from earlysign.schema.ES3.Base import MethodSpec as BaseMethodSpec
from earlysign.schema.ES3.GST.Boundary import StoppingRule


class AdaptationSpec(BaseModel):
    type: str


class SampleSizeReestimationSpec(AdaptationSpec):
    type: Literal["sample_size_reestimation"] = "sample_size_reestimation"
    method: Literal["conditional_power", "predictive_power"]
    target_power: float
    n_range: List[int]


class MethodSpec(BaseMethodSpec):
    kind: Literal["group_sequential"] = "group_sequential"

    efficacy: Optional[StoppingRule] = None
    futility: Optional[StoppingRule] = None

    adaptation: Optional[AdaptationSpec] = None
