from pydantic import Field

from earlysign.schema.ES3.base import BaseES3Model, Log


class BinomialArmData(Log):
    """
    Raw evidence: data for a single arm (Binomial).
    """

    total: int = Field(..., description="Total number of trials.")
    success: int = Field(..., description="Total number of successes.")
    arm: str = Field(..., description="The name/id of the arm.")


class ArmMetrics(BaseES3Model):
    """
    Standard statistics for a single arm (Bernoulli/Binomial).
    """

    total: int = Field(..., description="Total number of trials.")
    successes: int = Field(..., description="Total number of successes.")
    p_hat: float = Field(..., description="Success probability estimate (p-hat).")


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
