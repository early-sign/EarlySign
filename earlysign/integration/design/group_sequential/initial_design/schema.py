"""Design payload models for group sequential initial designs."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from earlysign.applications.design.group_sequential.initial_design.constants import (
    BindingMode,
    BoundaryScale,
    EfficacyStyle,
    FutilityMode,
    HypothesisStructure,
    SpendingFamily,
    StatisticType,
    normalize_spending_family,
)


class EfficacySpec(BaseModel):
    """Efficacy (upper) boundary configuration."""

    style: EfficacyStyle
    family: Optional[SpendingFamily] = None
    gamma: Optional[float] = None
    alpha_levels: Optional[Union[Sequence[float], Mapping[int, float]]] = None
    fixed_thresholds: Optional[Union[float, Mapping[int, float]]] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("family", mode="before")
    @classmethod
    def _coerce_family(
        cls, value: Optional[Union[SpendingFamily, str]]
    ) -> Optional[SpendingFamily]:
        return normalize_spending_family(value)

    @model_validator(mode="after")
    def _validate_configuration(self) -> "EfficacySpec":
        if self.style == EfficacyStyle.SIGNIFICANCE_LEVEL and self.alpha_levels is None:
            raise ValueError("significance_level style requires alpha_levels.")
        if self.style == EfficacyStyle.ALPHA_SPENDING and self.family is None:
            object.__setattr__(self, "family", SpendingFamily.HSD)
        if (
            self.style == EfficacyStyle.FIXED_THRESHOLD
            and self.fixed_thresholds is None
        ):
            raise ValueError("fixed_threshold style requires fixed_thresholds.")
        return self

    def resolved_alpha_levels(self) -> Optional[Dict[int, float]]:
        """Return alpha levels indexed by look if available."""

        if self.alpha_levels is None:
            return None
        if isinstance(self.alpha_levels, Mapping):
            return {int(k): float(v) for k, v in self.alpha_levels.items()}
        return {i + 1: float(v) for i, v in enumerate(self.alpha_levels)}


class FutilitySpec(BaseModel):
    """Futility (lower) boundary configuration."""

    mode: FutilityMode
    binding_mode: BindingMode = BindingMode.NON_BINDING
    z: Optional[Union[float, Mapping[int, float]]] = None
    family: Optional[SpendingFamily] = None
    gamma: Optional[float] = None
    beta: Optional[float] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("family", mode="before")
    @classmethod
    def _coerce_family(
        cls, value: Optional[Union[SpendingFamily, str]]
    ) -> Optional[SpendingFamily]:
        return normalize_spending_family(value)

    @model_validator(mode="after")
    def _validate_configuration(self) -> "FutilitySpec":
        if self.mode == FutilityMode.FIXED_THRESHOLD and self.z is None:
            raise ValueError("fixed_threshold mode requires z thresholds.")
        if self.mode == FutilityMode.BETA_SPENDING and self.family is None:
            object.__setattr__(self, "family", SpendingFamily.HSD)
        return self

    @property
    def is_binding(self) -> bool:
        return self.binding_mode is BindingMode.BINDING


class StatisticSpec(BaseModel):
    """Specification of the test statistic used in the design."""

    kind: StatisticType = Field(default=StatisticType.WALD_Z)
    scale: BoundaryScale = Field(default=BoundaryScale.Z)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class HypothesisSpec(BaseModel):
    """Hypothesis geometry (tails/sidedness)."""

    structure: HypothesisStructure = HypothesisStructure.TWO_SIDED_SYMMETRIC
    tail: Optional[str] = Field(
        default=None,
        description="Direction for one-sided tests ('upper' or 'lower').",
    )
    description: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_tail(self) -> "HypothesisSpec":
        if self.structure in (
            HypothesisStructure.ONE_SIDED_UPPER,
            HypothesisStructure.ONE_SIDED_LOWER,
        ):
            if self.tail is None:
                # Default tail direction to match structure
                default_tail = (
                    "upper"
                    if self.structure is HypothesisStructure.ONE_SIDED_UPPER
                    else "lower"
                )
                object.__setattr__(self, "tail", default_tail)
        else:
            object.__setattr__(self, "tail", None)
        return self

    @property
    def tails(self) -> int:
        if self.structure in (
            HypothesisStructure.ONE_SIDED_UPPER,
            HypothesisStructure.ONE_SIDED_LOWER,
        ):
            return 1
        return 2


class DesignPayloadModel(BaseModel):
    """Canonical representation of a group sequential design payload."""

    alpha: float = Field(..., gt=0.0, lt=1.0)
    hypothesis: HypothesisSpec = Field(default_factory=HypothesisSpec)
    statistic: StatisticSpec = Field(default_factory=StatisticSpec)
    efficacy: EfficacySpec = Field(
        default_factory=lambda: EfficacySpec(style=EfficacyStyle.ALPHA_SPENDING)
    )
    futility: FutilitySpec = Field(
        default_factory=lambda: FutilitySpec(mode=FutilityMode.NONE)
    )
    planned_max_n: int = Field(..., gt=0)
    planned_info_times: List[float] = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("planned_info_times")
    @classmethod
    def _validate_info_times(cls, value: List[float]) -> List[float]:
        prev = 0.0
        normalised: List[float] = []
        for idx, info in enumerate(value, start=1):
            fval = float(info)
            if not 0.0 < fval <= 1.0:
                raise ValueError(
                    f"planned_info_times[{idx}] must lie in (0, 1], got {info}."
                )
            if fval < prev:
                raise ValueError("planned_info_times must be non-decreasing.")
            prev = fval
            normalised.append(fval)
        return normalised

    def to_payload(self) -> Dict[str, Any]:
        """Normalise output to primitive python types ready for ledger storage."""

        return self.model_dump(mode="json", exclude_none=True)

    def boundary_spec(self) -> Dict[str, Any]:
        """Produce the legacy boundary-calculator payload."""

        futility_payload = self.futility.model_dump(
            mode="json", exclude_none=True, exclude={"binding_mode"}
        )
        futility_payload["binding"] = self.futility.is_binding

        return {
            "alpha": float(self.alpha),
            "tails": self.hypothesis.tails,
            "scale": self.statistic.scale.value,
            "statistic": self.statistic.kind.value,
            "efficacy": self.efficacy.model_dump(mode="json", exclude_none=True),
            "futility": futility_payload,
            "binding_mode": self.futility.binding_mode.value,
        }
