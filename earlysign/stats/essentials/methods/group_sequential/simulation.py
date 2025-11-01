"""Sampling helpers for group-sequential simulations."""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple


def compute_cumulative_sample_sizes(
    info_times: Sequence[float], planned_max_n: int
) -> List[int]:
    """Return cumulative sample-size targets using ceiling rounding."""

    max_total = int(planned_max_n)
    if max_total <= 0:
        raise ValueError("planned_max_n must be positive")

    rates = [float(rate) for rate in info_times]
    if not rates:
        raise ValueError("info_times must be non-empty")

    cumulative: List[int] = []
    previous = 0
    for idx, rate in enumerate(rates):
        target = previous if rate <= 0 else int(ceil(max_total * rate))
        if idx == len(rates) - 1:
            target = max_total
        target = max(previous, min(target, max_total))
        cumulative.append(target)
        previous = target

    cumulative[-1] = max_total
    return cumulative


class SamplingStrategy(Protocol):
    """Iterator-style interface describing sampling increments."""

    @property
    def description(self) -> str: ...

    @property
    def batch_size(self) -> Optional[int]: ...

    @property
    def schedule(self) -> Sequence[Tuple[int, int]]: ...

    @property
    def max_total(self) -> int: ...

    def metadata(self) -> Dict[str, Any]: ...

    def __iter__(self) -> Iterator[Tuple[int, int]]: ...


class _BaseSamplingStrategy:
    """Shared behaviour for concrete sampling strategies."""

    _description: str = "sampling"

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        return iter(self.schedule)

    @property
    def description(self) -> str:
        return self._description

    @property
    def batch_size(self) -> Optional[int]:  # pragma: no cover - overridden
        return None

    @property
    def schedule(self) -> Sequence[Tuple[int, int]]:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def max_total(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True)
class InfoTimeSampling(_BaseSamplingStrategy):
    """Sampling aligned to information-time fractions."""

    info_times: Sequence[float]
    planned_max_n: int
    allocation_ratio: float
    _cumulative: Tuple[int, ...] = field(init=False, repr=False)
    _schedule: Tuple[Tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        alloc = float(self.allocation_ratio)
        if alloc <= 0.0:
            raise ValueError("allocation_ratio must be positive")

        cumulative = compute_cumulative_sample_sizes(
            self.info_times, int(self.planned_max_n)
        )
        schedule: List[Tuple[int, int]] = []
        prev_a = 0
        prev_b = 0
        for total in cumulative:
            target_a = int(ceil(total / (1.0 + alloc)))
            target_b = total - target_a
            inc_a = max(0, target_a - prev_a)
            inc_b = max(0, target_b - prev_b)
            schedule.append((inc_a, inc_b))
            prev_a = target_a
            prev_b = target_b

        object.__setattr__(self, "_cumulative", tuple(int(x) for x in cumulative))
        object.__setattr__(
            self, "_schedule", tuple((int(a), int(b)) for a, b in schedule)
        )
        object.__setattr__(self, "_description", "info_times")

    @property
    def schedule(self) -> Sequence[Tuple[int, int]]:
        return self._schedule

    @property
    def max_total(self) -> int:
        return int(self._cumulative[-1])

    def metadata(self) -> Dict[str, Any]:
        return {
            "info_times": [float(x) for x in self.info_times],
            "planned_max_n": int(self.planned_max_n),
            "allocation_ratio": float(self.allocation_ratio),
            "cumulative_sizes": [int(x) for x in self._cumulative],
        }


@dataclass(frozen=True)
class FixedBatchSampling(_BaseSamplingStrategy):
    """Sampling with constant batch sizes up to a maximum total."""

    size: int
    allocation_ratio: float
    total: int
    _cumulative: Tuple[int, ...] = field(init=False, repr=False)
    _schedule: Tuple[Tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.total) <= 0:
            raise ValueError("max_total must be positive")

        alloc = float(self.allocation_ratio)
        if alloc <= 0.0:
            raise ValueError("allocation_ratio must be positive")

        schedule: List[Tuple[int, int]] = []
        cumulative: List[int] = []
        total = 0
        batch_total = int(self.size + round(self.size * alloc))
        if batch_total <= 0:
            raise ValueError("Derived batch total must be positive")

        total_a = 0
        total_b = 0
        while total < int(self.total):
            remaining = int(self.total) - total
            cur_total = min(batch_total, remaining)
            inc_a = int(round(cur_total / (1.0 + alloc)))
            inc_a = max(0, min(inc_a, cur_total))
            inc_b = cur_total - inc_a
            if inc_a == 0 and inc_b == 0:
                raise ValueError("FixedBatchSampling produced a zero increment")
            schedule.append((inc_a, inc_b))
            total_a += inc_a
            total_b += inc_b
            total = total_a + total_b
            cumulative.append(total)

        object.__setattr__(
            self, "_schedule", tuple((int(a), int(b)) for a, b in schedule)
        )
        object.__setattr__(self, "_cumulative", tuple(int(x) for x in cumulative))
        object.__setattr__(self, "_description", "batch")

    @property
    def batch_size(self) -> Optional[int]:
        return int(self.size)

    @property
    def schedule(self) -> Sequence[Tuple[int, int]]:
        return self._schedule

    @property
    def max_total(self) -> int:
        return int(self._cumulative[-1])

    def metadata(self) -> Dict[str, Any]:
        return {
            "batch_size": int(self.size),
            "allocation_ratio": float(self.allocation_ratio),
            "cumulative_sizes": [int(x) for x in self._cumulative],
            "max_total": int(self.total),
        }
