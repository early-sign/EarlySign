"""Design payload models and spec helpers for group sequential initial designs."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from earlysign.v0.methods.group_sequential.design.initial_design.constants import (
    SpendingFamily,
)
from earlysign.v0.stats.schemes.survival.effect_size import (
    TimeToEventEffect,
    TimeToEventSampleSize,
)
from earlysign.v0.stats.schemes.two_means.effect_size import MeansEffect, MeansSampleSize
from earlysign.v0.stats.schemes.two_proportions.effect_size import (
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
class SpendingFamilySpec:
    """Spending family selection with family-specific parameters."""

    family: SpendingFamily = SpendingFamily.OBF
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryConfig:
    """Efficacy and futility boundary configuration."""

    spending: SpendingFamilySpec = field(
        default_factory=lambda: SpendingFamilySpec(family=SpendingFamily.OBF)
    )
    futility_enabled: bool = False
    futility_spending: Optional[SpendingFamilySpec] = None
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
        """Return normalized information times for the configured number of analyses."""

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
        """Serialize the design spec to JSON."""

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
