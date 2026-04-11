from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from typing_extensions import TypeAliasType

from earlysign.builtin.group_sequential.schema.enums import (
    MethodModel,
    VarianceEstimation,
    VarianceSource,
)


class KnownVariance(BaseModel):
    kind: Literal["known"] = "known"
    value: float


class OneArmEstimatedVariance(BaseModel):
    kind: Literal["estimated"] = "estimated"


OneArmContinuousVariance = TypeAliasType(
    "OneArmContinuousVariance", KnownVariance | OneArmEstimatedVariance
)


class TwoArmEstimatedVariance(BaseModel):
    kind: Literal["estimated"] = "estimated"
    method: MethodModel


TwoArmContinuousVariance = TypeAliasType(
    "TwoArmContinuousVariance", KnownVariance | TwoArmEstimatedVariance
)


class TestStatisticSpec(BaseModel):
    """
    The "Statistic": Operational Definition of the Test Statistic.
    """

    kind: str


class BinomialExactStatistic(TestStatisticSpec):
    kind: Literal["binomial_exact"] = "binomial_exact"


class OneArmBinomialZ(TestStatisticSpec):
    kind: Literal["one_arm_binomial_z"] = "one_arm_binomial_z"
    information_unit: Literal["fisher_information"] = "fisher_information"
    variance_source: VarianceSource


class OneArmContinuousZ(TestStatisticSpec):
    kind: Literal["one_arm_continuous_z"] = "one_arm_continuous_z"
    information_unit: Literal["fisher_information"] = "fisher_information"
    variance: OneArmContinuousVariance


class TwoArmBinomialZ(TestStatisticSpec):
    kind: Literal["two_arm_binomial_z"] = "two_arm_binomial_z"
    information_unit: Literal["fisher_information"] = "fisher_information"
    variance_estimation: VarianceEstimation


class TwoArmContinuousZ(TestStatisticSpec):
    kind: Literal["two_arm_continuous_z"] = "two_arm_continuous_z"
    information_unit: Literal["fisher_information"] = "fisher_information"
    variance: TwoArmContinuousVariance
