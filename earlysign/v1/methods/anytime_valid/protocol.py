from typing import Optional

from pydantic import BaseModel, Field


class EProcessProtocol(BaseModel):
    """
    Protocol for continuous monitoring based on E-processes.

    E-processes allow for anytime-valid testing, where a rejection at any point
    (without a fixed schedule) is scientifically valid.
    """

    alpha: float = Field(0.05, description="Type-1 error rate")
    null_p: float = Field(
        0.5, description="Success probability hypothesized under the Null (H0)."
    )
    alt_p: Optional[float] = Field(
        None, description="Success probability hypothesized under the Alternative (H1)."
    )
