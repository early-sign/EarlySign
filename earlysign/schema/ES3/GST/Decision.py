from enum import Enum


class DecisionStatus(str, Enum):
    """
    Standard decision statuses for GST and other designs.
    """

    CONTINUE = "CONTINUE"
    STOP_EFFICACY = "STOP_EFFICACY"
    STOP_FUTILITY = "STOP_FUTILITY"
    STOP_PLAN_END_REACHED = "STOP_PLAN_END_REACHED"
    STOP = "STOP"
