from typing import List, Optional

from pydantic import BaseModel, Field


class GSTProtocol(BaseModel):
    """
    Protocol for Group Sequential Tests (GST).

    This model serves a dual purpose:
    1. **Planning**: Defines the initial scientific intent (alpha, power, delta).
    2. **Realization**: Records the concrete design parameters (n_max, milestones, boundaries)
       once the design is resolved via a planner.
    """

    alpha: float = Field(
        0.05, description="Type-1 error rate (significance level) for the study."
    )
    power: float = Field(0.8, description="Target statistical power (1 - Beta).")
    K: int = Field(3, description="Total number of looks (interim + final).")
    spending_function: str = Field(
        "rho_family",
        description="Type of alpha spending function (e.g., 'rho_family', 'obrien_fleming').",
    )
    rho: float = Field(
        3.0, description="Rho parameter for the spending function (if applicable)."
    )
    side: int = Field(1, description="Number of sides for the test (1 or 2).")
    delta: Optional[float] = Field(
        None, description="Minimum clinically meaningful difference (target effect)."
    )

    # Design Realization Fields
    n_max: int = Field(
        0, description="Total maximum sample size (sum of both arms if AB)."
    )
    milestones: List[float] = Field(
        default_factory=list,
        description="List of information fractions (0 to 1) at which looks occur.",
    )
    boundaries: List[float] = Field(
        default_factory=list,
        description="List of critical values (Z-scores) corresponding to the milestones.",
    )
