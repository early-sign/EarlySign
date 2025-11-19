"""Design payload models and spec helpers for group sequential initial designs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from earlysign.integration.design.group_sequential.initial_design.constants import (
    BindingMode,
    BoundaryScale,
    EfficacyStyle,
    FutilityMode,
    HypothesisStructure,
    SpendingFamily,
    StatisticType,
    normalize_spending_family,
)
from earlysign.stats.schemes.survival.effect_size import (
    TimeToEventEffect,
    TimeToEventSampleSize,
)
from earlysign.stats.schemes.two_means.effect_size import (
    MeansEffect,
    MeansSampleSize,
)
from earlysign.stats.schemes.two_proportions.effect_size import (
    ProportionsEffect,
    ProportionsSampleSize,
)

# ---------------------------------------------------------------------------
# Lightweight config objects used by optimization workflows and UI helpers.
# ---------------------------------------------------------------------------


class SpendingFunction(str, Enum):
    """Alpha spending function types."""

    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    HSD = "hsd"


class TestType(str, Enum):
    """Statistical test types supported by design specs."""

    TWO_PROPORTIONS = "two_sample_proportions"
    TWO_MEANS = "two_sample_means"
    TIME_TO_EVENT = "time_to_event"


class TimingType(str, Enum):
    """Timing specification for interim analyses."""

    INFORMATION_BASED = "information_based"
    CALENDAR_TIME = "calendar_time"


class InformationSpacing(str, Enum):
    """Information time spacing strategies."""

    EQUAL = "equal"
    CUSTOM = "custom"


class DesignMode(str, Enum):
    """Design optimization modes supported by the notebook UI."""

    NMAX_FIXED_MIN_MDE = "nmax_fixed_min_mde"
    FIXED_TIMING = "fixed_timing"
    OPTIMIZE_ASN = "optimize_asn"
    OPTIMIZE_DESIGN = "optimize_design"
    FIXED_POWER = "fixed_power"


@dataclass
class TestConfig:
    """Test-specific configuration."""

    test_type: TestType = TestType.TWO_PROPORTIONS
    sided: Literal["one", "two"] = "one"
    alpha: float = 0.025
    power: float = 0.90


@dataclass
class SequentialConfig:
    """Sequential design configuration."""

    n_analyses: int = 3
    timing_type: TimingType = TimingType.INFORMATION_BASED
    info_spacing: InformationSpacing = InformationSpacing.EQUAL
    info_times: Optional[List[float]] = None

    def get_info_times(self, n_analyses: int) -> np.ndarray:
        """Return information times (custom or equally spaced)."""

        if self.info_times and self.info_spacing == InformationSpacing.CUSTOM:
            info = np.array(self.info_times, dtype=float)
        else:
            info = np.array(
                [(i + 1) / n_analyses for i in range(n_analyses)], dtype=float
            )
        info = np.clip(info, 1e-6, 1.0)
        info = np.maximum.accumulate(info)
        return info


@dataclass
class AllocationConfig:
    """Sample allocation configuration (n_treatment / n_control)."""

    alloc_ratio: float = 1.0


@dataclass
class BoundaryConfig:
    """Efficacy and futility boundary configuration."""

    spending_function: SpendingFunction = SpendingFunction.OBRIEN_FLEMING
    hsd_gamma: float = -4.0
    futility_enabled: bool = False
    futility_spending: Optional[SpendingFunction] = None
    futility_z: Optional[float] = None


@dataclass
class SimulationConfig:
    """Simulation parameters."""

    seed: Optional[int] = None
    n_sims: int = 2000


@dataclass
class DisplayConfig:
    """Display/formatting configuration."""

    digits: int = 4
    ddigits: int = 2
    tdigits: int = 1


@dataclass
class DesignSpec:
    """Hierarchical design specification for sequential trials."""

    test: TestConfig = field(default_factory=TestConfig)
    sequential: SequentialConfig = field(default_factory=SequentialConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    effect: Any = None
    sample_size: Any = None

    def resolved_info_times(self) -> np.ndarray:
        return self.sequential.get_info_times(self.sequential.n_analyses)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

        data = asdict(self)

        def convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        result = convert(data)
        assert isinstance(result, dict)
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class ProportionsDesignSpec(DesignSpec):
    """Design specification for two-sample proportions."""

    effect: ProportionsEffect = field(default_factory=ProportionsEffect)
    sample_size: ProportionsSampleSize = field(default_factory=ProportionsSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TWO_PROPORTIONS


@dataclass
class TimeToEventDesignSpec(DesignSpec):
    """Design specification for time-to-event analyses."""

    effect: TimeToEventEffect = field(default_factory=TimeToEventEffect)
    sample_size: TimeToEventSampleSize = field(default_factory=TimeToEventSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TIME_TO_EVENT


@dataclass
class MeansDesignSpec(DesignSpec):
    """Design specification for two-sample means."""

    effect: MeansEffect = field(default_factory=MeansEffect)
    sample_size: MeansSampleSize = field(default_factory=MeansSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TWO_MEANS


# ---------------------------------------------------------------------------
# Pydantic payload models used when persisting designs to the ledger.
# ---------------------------------------------------------------------------


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
