from typing import Optional

from pydantic import BaseModel, Field


class GSTProtocol(BaseModel):
    """
    Protocol for Group Sequential Tests.
    """

    alpha: float = Field(0.05, description="Type-1 error rate")
    power: float = Field(0.8, description="Target power")
    K: int = Field(3, description="Number of looks")
    spending_function: str = Field(
        "rho_family", description="Type of spending function"
    )
    rho: float = Field(3.0, description="Rho parameter for spending function")
    side: int = Field(1, description="1-sided or 2-sided test")
    delta: Optional[float] = Field(None, description="Target clinical difference")


class EProcessProtocol(BaseModel):
    """
    Protocol for e-process based monitoring.
    """

    alpha: float = Field(0.05, description="Type-1 error rate")
    null_p: float = Field(0.5, description="Hypothesized probability under H0")
    alt_p: Optional[float] = Field(
        None, description="Hypothesized probability under H1"
    )
