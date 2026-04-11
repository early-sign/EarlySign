from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from earlysign.builtin.group_sequential.schema.enums import (
    Method,
    MultiplicityAdjustment,
    ResponseType,
)
from earlysign.builtin.group_sequential.schema.hypotheses import HypothesisSpec
from earlysign.builtin.group_sequential.schema.logs import AdaptationSnapshot
from earlysign.builtin.group_sequential.schema.policies import StoppingPolicySpec
from earlysign.schema.ES3.base import (
    ArmStructure,
    MethodSpec as MethodSpec_1,
    Protocol as Protocol_1,
    TaskSpec as TaskSpec_1,
)


class EfficacyRequirement(BaseModel):
    alpha: float
    binding: bool | None = Field(True, description="Binding.")


class FutilityRequirement(BaseModel):
    power: float = Field(..., description="Target Power (1 - Beta)")
    binding: bool | None = Field(False, description="Non-binding.")


class PromisingZoneSpec(BaseModel):
    conditional_power_threshold_min: float
    conditional_power_threshold_max: float
    target_conditional_power: float


class SampleSizeReestimationSpec(BaseModel):
    type: Literal["sample_size_reestimation"] = "sample_size_reestimation"
    method: Method
    target_power: float
    use_weighted_statistic: bool | None = True
    n_range: list[Any]
    promising_zone: PromisingZoneSpec | None = None


class AdaptationSpec(BaseModel):
    sample_size_reestimation: SampleSizeReestimationSpec | None = None


class TaskSpec(TaskSpec_1):
    kind: Literal["group_sequential"] = "group_sequential"
    arms: ArmStructure
    response_type: ResponseType
    hypotheses: HypothesisSpec
    efficacy: EfficacyRequirement | None = None
    futility: FutilityRequirement | None = None


class MethodSpec(MethodSpec_1):
    kind: Literal["group_sequential"] = Field("group_sequential")
    stopping_policy: StoppingPolicySpec
    adaptation: AdaptationSpec | None = None
    adaptation_snapshot: AdaptationSnapshot | None = None


class Protocol(Protocol_1):
    task: TaskSpec
    method: MethodSpec


class MultivariateTaskSpec(TaskSpec_1):
    kind: Literal["group_sequential_multivariate"] = "group_sequential_multivariate"
    sub_tasks: list[TaskSpec]
    arms: list[str]
    hypotheses: HypothesisSpec
    alpha: float | None = None
    beta: float | None = None


class MultivariateMethodSpec(MethodSpec_1):
    kind: Literal["group_sequential_multivariate"] = "group_sequential_multivariate"
    multiplicity_adjustment: MultiplicityAdjustment
    correlation_matrix: list[list[float]] | None = None


class MultivariateProtocol(Protocol_1):
    task: MultivariateTaskSpec
    method: MultivariateMethodSpec
