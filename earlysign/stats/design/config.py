"""Configuration dataclasses for group sequential trial designs."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from earlysign.stats.design.types import (
    InformationSpacing,
    SpendingFunction,
    TestType,
    TimingType,
)


@dataclass
class TestConfig:
    """Test-specific configuration.

    >>> config = TestConfig(test_type=TestType.TWO_PROPORTIONS, alpha=0.025)
    >>> config.alpha
    0.025
    >>> config.sided
    'one'
    """

    test_type: TestType = TestType.TWO_PROPORTIONS
    sided: Literal["one", "two"] = "one"
    alpha: float = 0.025
    power: float = 0.90


@dataclass
class SequentialConfig:
    """Sequential design configuration.

    >>> config = SequentialConfig(n_analyses=3)
    >>> times = config.get_info_times(3)
    >>> np.allclose(times, [0.333333, 0.666667, 1.0], atol=0.001)
    True
    """

    n_analyses: int = 3
    timing_type: TimingType = TimingType.INFORMATION_BASED
    info_spacing: InformationSpacing = InformationSpacing.EQUAL
    info_times: Optional[List[float]] = None

    def get_info_times(self, n_analyses: int) -> np.ndarray:
        """Get information times (custom or equally spaced).

        Args:
            n_analyses: Number of analyses

        Returns:
            Array of information times in (0, 1]

        >>> config = SequentialConfig(info_times=[0.3, 0.7, 1.0],
        ...                          info_spacing=InformationSpacing.CUSTOM)
        >>> times = config.get_info_times(3)
        >>> np.allclose(times, [0.3, 0.7, 1.0])
        True
        """
        if self.info_times and self.info_spacing == InformationSpacing.CUSTOM:
            t = np.array(self.info_times, dtype=float)
        else:
            t = np.array([(i + 1) / n_analyses for i in range(n_analyses)], dtype=float)

        # Ensure monotonic increase and within (0, 1]
        t = np.clip(t, 1e-6, 1.0)
        t = np.maximum.accumulate(t)
        return t


@dataclass
class AllocationConfig:
    """Sample allocation configuration.

    >>> config = AllocationConfig(alloc_ratio=2.0)
    >>> config.alloc_ratio  # nB / nA (treatment / control)
    2.0
    """

    alloc_ratio: float = 1.0  # nB / nA (treatment / control)


@dataclass
class BoundaryConfig:
    """Efficacy and futility boundary configuration.

    >>> config = BoundaryConfig(spending_function=SpendingFunction.OBRIEN_FLEMING)
    >>> config.futility_enabled
    False
    >>> config.hsd_gamma
    -4.0
    """

    # Efficacy boundary
    spending_function: SpendingFunction = SpendingFunction.OBRIEN_FLEMING
    hsd_gamma: float = -4.0

    # Futility boundary
    futility_enabled: bool = False
    futility_spending: Optional[SpendingFunction] = None
    futility_z: Optional[float] = None


@dataclass
class SimulationConfig:
    """Simulation parameters.

    >>> config = SimulationConfig(seed=123, n_sims=10000)
    >>> config.seed
    123
    """

    seed: int = 42
    n_sims: int = 5000


@dataclass
class DisplayConfig:
    """Display and formatting configuration.

    >>> config = DisplayConfig(digits=3, ddigits=2)
    >>> config.digits
    3
    """

    digits: int = 4  # Significant figures
    ddigits: int = 2  # Decimal places
    tdigits: int = 1  # Time display precision


# ========== Effect Specifications ==========


@dataclass
class ProportionsEffect:
    """Effect specification for two-sample proportions test.

    >>> effect = ProportionsEffect(p_control=0.10, effect_size=0.02)
    >>> round(effect.get_treatment_proportion(), 10)
    0.12
    >>> effect_rel = ProportionsEffect(p_control=0.10, effect_size=0.2,
    ...                                effect_type="relative")
    >>> effect_rel.get_treatment_proportion()
    0.12
    """

    p_control: float = 0.10
    p_treatment: Optional[float] = None
    effect_size: Optional[float] = 0.02
    effect_type: Literal["absolute", "relative", "odds_ratio"] = "absolute"

    def get_treatment_proportion(self) -> float:
        """Calculate treatment proportion based on effect specification.

        Returns:
            Treatment group proportion

        Raises:
            ValueError: If effect_type is unknown or effect_size is None
        """
        if self.p_treatment is not None:
            return self.p_treatment

        if self.effect_size is None:
            raise ValueError("effect_size must be specified when p_treatment is None")

        if self.effect_type == "absolute":
            return self.p_control + self.effect_size
        elif self.effect_type == "relative":
            return self.p_control * (1 + self.effect_size)
        elif self.effect_type == "odds_ratio":
            odds_control = self.p_control / (1 - self.p_control)
            odds_treatment = odds_control * self.effect_size
            return odds_treatment / (1 + odds_treatment)
        else:
            raise ValueError(f"Unknown effect_type: {self.effect_type}")


@dataclass
class TimeToEventEffect:
    """Effect specification for time-to-event analysis.

    >>> effect = TimeToEventEffect(hazard_ratio=0.7)
    >>> effect.hazard_ratio
    0.7
    """

    hazard_ratio: float = 0.6


@dataclass
class MeansEffect:
    """Effect specification for two-sample means test.

    >>> effect = MeansEffect(mean_control=10.0, effect_size=2.0, std_dev=5.0)
    >>> effect.mean_control
    10.0
    >>> effect.std_dev
    5.0
    """

    mean_control: float = 10.0
    mean_treatment: Optional[float] = None
    effect_size: Optional[float] = 2.0
    std_dev: float = 5.0
    pooled_std: bool = True


# ========== Sample Size Specifications ==========


@dataclass
class ProportionsSampleSize:
    """Sample size specification for proportions test.

    >>> sample_size = ProportionsSampleSize(n_per_analysis=500)
    >>> sample_size.n_per_analysis
    500
    """

    n_per_analysis: int = 500  # Per group


@dataclass
class TimeToEventSampleSize:
    """Sample size specification for time-to-event analysis.

    >>> sample_size = TimeToEventSampleSize(total_events=171,
    ...                                     total_sample_size=296)
    >>> sample_size.total_events
    171
    """

    total_events: int = 171
    total_sample_size: int = 296
    time_unit: Literal["days", "weeks", "months", "years"] = "months"
    accrual_duration: Optional[float] = None
    follow_up_duration: Optional[float] = None


@dataclass
class MeansSampleSize:
    """Sample size specification for means test.

    >>> sample_size = MeansSampleSize(n_per_analysis=100)
    >>> sample_size.n_per_analysis
    100
    """

    n_per_analysis: int = 100  # Per group


# ========== Main Design Specifications ==========


@dataclass
class DesignSpec:
    """Hierarchical design specification for sequential trials.

    Organized hierarchical structure:
    - test: Test configuration (type, significance level, power)
    - sequential: Sequential design configuration (# analyses, information times)
    - allocation: Sample allocation configuration
    - boundary: Boundary configuration (efficacy and futility)
    - simulation: Simulation configuration
    - display: Display configuration
    - effect: Effect size specification (test-type specific)
    - sample_size: Sample size specification (test-type specific)

    >>> spec = DesignSpec()
    >>> spec.test.alpha
    0.025
    >>> times = spec.resolved_info_times()
    >>> len(times)
    3
    """

    test: TestConfig = field(default_factory=TestConfig)
    sequential: SequentialConfig = field(default_factory=SequentialConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # Test-type specific configurations (set based on test_type)
    effect: Any = None
    sample_size: Any = None

    def resolved_info_times(self) -> np.ndarray:
        """Get resolved information times.

        Returns:
            Array of information times
        """
        return self.sequential.get_info_times(self.sequential.n_analyses)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with enums converted to strings

        >>> spec = DesignSpec()
        >>> d = spec.to_dict()
        >>> d['test']['alpha']
        0.025
        """
        d = asdict(self)

        # Convert Enums to strings
        def convert_enums(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        result = convert_enums(d)
        assert isinstance(result, dict)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class ProportionsDesignSpec(DesignSpec):
    """Design specification for two-sample proportions test.

    >>> spec = ProportionsDesignSpec()
    >>> spec.test.test_type == TestType.TWO_PROPORTIONS
    True
    >>> spec.effect.p_control
    0.1
    """

    effect: ProportionsEffect = field(default_factory=ProportionsEffect)
    sample_size: ProportionsSampleSize = field(default_factory=ProportionsSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TWO_PROPORTIONS


@dataclass
class TimeToEventDesignSpec(DesignSpec):
    """Design specification for time-to-event analysis.

    >>> spec = TimeToEventDesignSpec()
    >>> spec.test.test_type == TestType.TIME_TO_EVENT
    True
    >>> spec.effect.hazard_ratio
    0.6
    """

    effect: TimeToEventEffect = field(default_factory=TimeToEventEffect)
    sample_size: TimeToEventSampleSize = field(default_factory=TimeToEventSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TIME_TO_EVENT


@dataclass
class MeansDesignSpec(DesignSpec):
    """Design specification for two-sample means test.

    >>> spec = MeansDesignSpec()
    >>> spec.test.test_type == TestType.TWO_MEANS
    True
    >>> spec.effect.std_dev
    5.0
    """

    effect: MeansEffect = field(default_factory=MeansEffect)
    sample_size: MeansSampleSize = field(default_factory=MeansSampleSize)

    def __post_init__(self) -> None:
        self.test.test_type = TestType.TWO_MEANS
