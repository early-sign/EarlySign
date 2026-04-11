from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypeAliasType

from earlysign.builtin.group_sequential.schema.enums import Sided, SpendingFunctionType


class StatisticalModel(BaseModel):
    kind: str


class CanonicalGaussianModel(StatisticalModel):
    kind: Literal["canonical_gaussian"] = "canonical_gaussian"


class ExactBinomialModel(StatisticalModel):
    kind: Literal["exact_binomial"] = "exact_binomial"


class TProcessModel(StatisticalModel):
    kind: Literal["t_process"] = "t_process"
    degrees_of_freedom_method: str


class DecisionStrategyBase(BaseModel):
    kind: str
    statistical_model: StatisticalModel


class BoundaryFunctionStrategyBase(DecisionStrategyBase):
    """Abstract Base for Shape-Based Boundary Strategies."""


class OBrienFlemingStrategy(BoundaryFunctionStrategyBase):
    kind: Literal["obrien_fleming"] = "obrien_fleming"
    alpha: float
    sided: Sided


class PocockStrategy(BoundaryFunctionStrategyBase):
    kind: Literal["pocock"] = "pocock"
    alpha: float
    sided: Sided


class WangTsiatisStrategy(BoundaryFunctionStrategyBase):
    kind: Literal["wang_tsiatis"] = "wang_tsiatis"
    alpha: float
    delta: float
    sided: Sided


class WhiteheadStrategy(BoundaryFunctionStrategyBase):
    kind: Literal["whitehead"] = "whitehead"
    alpha: float
    beta: float


class SpendingFunction(BaseModel):
    family: SpendingFunctionType | str
    params: dict[str, float] | None = None


class SpendingFunctionStrategyBase(DecisionStrategyBase):
    """Abstract Base for Spending-Function Strategies."""


class AlphaSpendingStrategy(SpendingFunctionStrategyBase):
    kind: Literal["alpha_spending"] = "alpha_spending"
    spending_fn: SpendingFunction
    budget: float
    sided: Sided


class BetaSpendingStrategy(SpendingFunctionStrategyBase):
    kind: Literal["beta_spending"] = "beta_spending"
    spending_fn: SpendingFunction
    budget: float


class AlphaBetaSpendingStrategy(SpendingFunctionStrategyBase):
    kind: Literal["alpha_beta_spending"] = "alpha_beta_spending"
    alpha_spending_fn: SpendingFunction
    beta_spending_fn: SpendingFunction
    alpha_budget: float
    beta_budget: float
    alpha_binding: bool | None = True
    beta_binding: bool | None = False


DecisionStrategy = TypeAliasType(
    "DecisionStrategy",
    Annotated[
        AlphaSpendingStrategy
        | BetaSpendingStrategy
        | AlphaBetaSpendingStrategy
        | WhiteheadStrategy
        | OBrienFlemingStrategy
        | PocockStrategy
        | WangTsiatisStrategy,
        Field(..., description="Discriminated Union for Decision Strategies."),
    ],
)
