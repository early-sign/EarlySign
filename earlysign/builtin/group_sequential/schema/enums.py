from enum import StrEnum


class DecisionStatus(StrEnum):
    """
    Status of an ongoing or completed Group Sequential Test.
    """

    CONTINUE = "continue"
    STOP_EFFICACY = "stop_efficacy"
    STOP_FUTILITY = "stop_futility"
    STOP_PLAN_END_REACHED = "stop_plan_end_reached"
    STOP = "stop"


class PromisingZoneStatus(StrEnum):
    FUTILITY = "futility"
    EFFICACY = "efficacy"
    PROMISING = "promising"
    CONTINUE = "continue"


class ResponseType(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    TIME_TO_EVENT = "time_to_event"


class Sided(StrEnum):
    ONE = "one"
    TWO = "two"


class Method(StrEnum):
    CONDITIONAL_POWER = "conditional_power"
    PREDICTIVE_POWER = "predictive_power"


class MethodModel(StrEnum):
    """
    - "pooled": Equal variance assumption.
    - "unpooled": Unequal variance (Welch).
    """

    POOLED = "pooled"
    UNPOOLED = "unpooled"


class MultiplicityAdjustment(StrEnum):
    NONE = "none"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    HOCHBERG = "hochberg"
    GATEKEEPING = "gatekeeping"


class SpendingFunctionType(StrEnum):
    """
    Available spending function types for group sequential design.
    """

    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    POWER_FAMILY = "power_family"
    HWANG_SHIH_DECANI = "hwang_shih_decani"


class Unit(StrEnum):
    """
    "individuals": Count individuals (e.g. n1 + n2).
    "effective_size": Effective sample size (e.g. 4*n1*n2/(n1+n2)).
    """

    INDIVIDUALS = "individuals"
    EFFECTIVE_SIZE = "effective_size"


class VarianceEstimation(StrEnum):
    """
    Variance estimation method.
    - "unpooled": Wald Z (Observed).
    - "pooled": Score Z (Null/Common).
    """

    POOLED = "pooled"
    UNPOOLED = "unpooled"


class VarianceSource(StrEnum):
    """
    Variance source.
    - "sample": Uses observed estimate (Wald-like).
    - "null_hypothesis": Uses null parameter (Score-like).
    """

    SAMPLE = "sample"
    NULL_HYPOTHESIS = "null_hypothesis"
