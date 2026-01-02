from typing import Optional

from pydantic import BaseModel, Field


class EProcessProtocol(BaseModel):
    """
    Protocol for e-process based monitoring.
    """

    alpha: float = Field(0.05, description="Type-1 error rate")
    null_p: float = Field(0.5, description="Hypothesized probability under H0")
    alt_p: Optional[float] = Field(
        None, description="Hypothesized probability under H1"
    )
