"""Group sequential design records and payload models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.integration.design.group_sequential.initial_design.constants import (
    BindingMode,
    BoundaryScale,
    EfficacyStyle,
    FutilityMode,
    HypothesisStructure,
    SpendingFamily,
    StatisticType,
)

AlphaLevels = Sequence[float] | Mapping[int, float]
Thresholds = float | Mapping[int, float]


class EfficacySpec(BaseModel):
    """
    Efficacy (upper) boundary configuration.

    Cross-field validation enforces:
    - significance_level style requires alpha_levels
    - fixed_threshold style requires fixed_thresholds
    - alpha_spending style backfills default family/params
    """

    style: EfficacyStyle
    family: Optional[SpendingFamily] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    alpha_levels: Optional[AlphaLevels] = None
    fixed_thresholds: Optional[Thresholds] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_configuration(self) -> "EfficacySpec":
        """Enforce cross-field rules and backfill defaults for spending setups."""

        updates: Dict[str, Any] = {}
        if self.style == EfficacyStyle.SIGNIFICANCE_LEVEL and self.alpha_levels is None:
            raise ValueError("significance_level style requires alpha_levels.")
        if self.style == EfficacyStyle.ALPHA_SPENDING:
            family = self.family or SpendingFamily.HSD
            updates["family"] = family
            if not self.params and family is SpendingFamily.HSD:
                updates["params"] = {"gamma": -4.0}
        if (
            self.style == EfficacyStyle.FIXED_THRESHOLD
            and self.fixed_thresholds is None
        ):
            raise ValueError("fixed_threshold style requires fixed_thresholds.")
        return self.model_copy(update=updates) if updates else self

    def resolved_alpha_levels(self) -> Optional[Dict[int, float]]:
        """Return alpha levels indexed by look if available."""

        if self.alpha_levels is None:
            return None
        if isinstance(self.alpha_levels, Mapping):
            return {int(k): float(v) for k, v in self.alpha_levels.items()}
        return {i + 1: float(v) for i, v in enumerate(self.alpha_levels)}


class FutilitySpec(BaseModel):
    """
    Futility (lower) boundary configuration.

    Cross-field validation enforces:
    - fixed_threshold mode requires z thresholds
    - beta_spending mode requires a beta param and sets a default family
    """

    mode: FutilityMode
    binding_mode: BindingMode = BindingMode.NON_BINDING
    z: Optional[Thresholds] = None
    family: Optional[SpendingFamily] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_configuration(self) -> "FutilitySpec":
        """Enforce cross-field rules and required params for futility modes."""

        updates: Dict[str, Any] = {}
        if self.mode == FutilityMode.FIXED_THRESHOLD and self.z is None:
            raise ValueError("fixed_threshold mode requires z thresholds.")
        if self.mode == FutilityMode.BETA_SPENDING:
            family = self.family or SpendingFamily.HSD
            updates["family"] = family
            if "beta" not in self.params:
                raise ValueError("beta_spending mode requires params['beta'].")
        return self.model_copy(update=updates) if updates else self

    @property
    def is_binding(self) -> bool:
        """Return True when futility stopping is binding."""
        return self.binding_mode is BindingMode.BINDING


class StatisticSpec(BaseModel):
    """Specification of the test statistic used in the design."""

    kind: StatisticType = Field(default=StatisticType.WALD_Z)
    scale: BoundaryScale = Field(default=BoundaryScale.Z)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class HypothesisSpec(BaseModel):
    """
    Hypothesis geometry (tails/sidedness).

    Validation normalises the tail based on the selected structure.
    """

    structure: HypothesisStructure = HypothesisStructure.TWO_SIDED_SYMMETRIC
    tail: Optional[str] = Field(
        default=None,
        description="Direction for one-sided tests ('upper' or 'lower').",
    )
    description: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_tail(self) -> "HypothesisSpec":
        updates: Dict[str, Any] = {}
        if self.structure in (
            HypothesisStructure.ONE_SIDED_UPPER,
            HypothesisStructure.ONE_SIDED_LOWER,
        ):
            if self.tail is None:
                default_tail = (
                    "upper"
                    if self.structure is HypothesisStructure.ONE_SIDED_UPPER
                    else "lower"
                )
                updates["tail"] = default_tail
        else:
            updates["tail"] = None
        return self.model_copy(update=updates) if updates else self

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
        """Ensure planned information times are in (0, 1] and non-decreasing."""

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
        """
        Normalise output to primitive python types ready for ledger storage.

        Empty mappings are stripped so JSON columns (e.g., DuckDB) do not receive
        empty structs that some backends reject.
        """

        payload = self.model_dump(mode="json", exclude_none=True)

        def _strip(value: Any) -> Any:
            if isinstance(value, dict):
                cleaned = {k: _strip(v) for k, v in value.items() if v is not None}
                return {k: v for k, v in cleaned.items() if v != {}}
            if isinstance(value, list):
                return [_strip(v) for v in value]
            return value

        return cast(Dict[str, Any], _strip(payload))

    def boundary_spec(self) -> Dict[str, Any]:
        """
        Produce the legacy boundary-calculator payload.

        This maintains compatibility with the existing boundary calculator while
        the rest of the stack transitions to the richer design spec.
        """

        futility_payload = self.futility.model_dump(
            mode="json", exclude_none=True, exclude={"binding_mode"}
        )
        futility_payload["binding"] = self.futility.is_binding

        if not self.futility.params:
            futility_payload.pop("params", None)

        return {
            "alpha": float(self.alpha),
            "tails": self.hypothesis.tails,
            "scale": self.statistic.scale.value,
            "statistic": self.statistic.kind.value,
            "efficacy": {
                k: v
                for k, v in self.efficacy.model_dump(
                    mode="json", exclude_none=True
                ).items()
                if not (k == "params" and v == {})
            },
            "futility": futility_payload,
            "binding_mode": self.futility.binding_mode.value,
        }


@dataclass(frozen=True)
class DesignPayload:
    """Container that bundles the metadata and pydantic payload."""

    spec: DesignPayloadModel
    metadata: Mapping[str, Any]


class GroupSequentialDesignRecord(LedgerRecord, QueryMixin):
    """Persisted design snapshots for group sequential testing."""

    schema: dict[str, object] = {}

    def _validate_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Validate against the design model and drop empty structures.

        Using the Pydantic model here ensures cross-field validation and keeps
        the payload compatible with JSON columns across connectors.
        """

        parsed = self.schema_pydantic_model.model_validate(dict(payload))
        if hasattr(parsed, "to_payload"):
            return parsed.to_payload()
        return parsed.model_dump(mode="json", exclude_none=True)

    @property
    def schema_pydantic_model(self) -> Type[DesignPayloadModel]:
        """Use the canonical design payload model for validation."""

        return DesignPayloadModel
