from pydantic import Field

from earlysign.schema.ES3.base import BaseES3Model, Log


class Observation(Log):
    """
    Raw evidence: data for a single arm (continuous observation).
    """

    value: float = Field(..., description="The observed value.")
    arm: str = Field(..., description="Arm identifier.")


class ContinuousArmData(Log):
    """
    Raw evidence: aggregated data for a single arm (continuous).
    """

    total: int = Field(..., description="Sample size.")
    sum_x: float = Field(..., description="Sum of observations.")
    sum_x2: float = Field(..., description="Sum of squared observations.")
    arm: str = Field(..., description="Arm identifier.")


class ArmMetrics(BaseES3Model):
    """
    Standard statistics for a single arm (Continuous).
    """

    total: int
    mean: float
    variance: float


class ArmStatus(BaseES3Model):
    """
    Latest status and metrics for a single arm.
    """

    metrics: ArmMetrics
    is_active: bool = Field(True, description="Whether the arm is still active.")


class Scoreboard(BaseES3Model):
    """
    Collective state of all arms (Scoreboard).
    """

    arms: dict[str, ArmStatus] = Field(..., description="Map of arm IDs to status.")
