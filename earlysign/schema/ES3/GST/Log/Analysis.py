from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel


class DecisionStatus(str, Enum):
    """
    Standard decision statuses for GST and other designs.
    """

    CONTINUE = "CONTINUE"
    STOP_EFFICACY = "STOP_EFFICACY"
    STOP_FUTILITY = "STOP_FUTILITY"
    STOP_PLAN_END_REACHED = "STOP_PLAN_END_REACHED"
    STOP = "STOP"
    COMPLETED = "COMPLETED"


class Trigger(BaseModel):
    """
    Defines the reason / cause for an operational event (e.g., Analysis Execution).
    """

    kind: str


class ScheduleTrigger(Trigger):
    """
    Triggered by planned protocol milestones or schedules.
    """

    kind: Literal["schedule"] = "schedule"
    index: int
    value: float


class ManualTrigger(Trigger):
    """
    Triggered manually by a human operator.
    """

    kind: Literal["manual"] = "manual"
    reason: str


class Analysis(BaseModel):
    """
    Structural Record of a GST Analysis Execution.
    Captured context, metadata and factual results.
    """

    trigger: Optional[Union[ScheduleTrigger, ManualTrigger, Trigger]] = None

    # Results
    look: int
    statistic: float
    info_frac: float
    decision: DecisionStatus

    efficacy_boundary: Optional[float] = None
    futility_boundary: Optional[float] = None


class Decision(BaseModel):
    """
    Formalized Decision Event.
    """

    status: DecisionStatus
    message: str
