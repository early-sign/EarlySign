"""
Procedure builders for two-proportions group-sequential simulations.

These helpers produce ``Procedure`` instances compatible with the simulation
utilities under ``earlysign.stats.essentials`` without relying on higher-level
application modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, SupportsInt

import numpy as np

from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)
from earlysign.stats.essentials.methods.group_sequential.spending import (
    SpendingFunction,
)
from earlysign.stats.essentials.schemes.two_proportions.wald_z import compute_wald_z

ProcedureFactory = Callable[
    [Sequence[float], int, Optional[Mapping[str, object]], Optional[int]],
    "TwoProportionsProcedure",
]


def build_two_prop_procedure_factory(
    *,
    spending_obj: SpendingFunction,
    alpha: float,
    allocation_ratio: float,
) -> ProcedureFactory:
    """Return a :class:`ProcedureFactory` tailored to two-proportion tests."""

    family = spending_obj.name

    def _factory(
        info_times: Sequence[float],
        planned_max_n: int,
        design_payload: Optional[Mapping[str, object]],
        rng_seed: Optional[int],
    ) -> TwoProportionsProcedure:
        rates = np.asarray(info_times, dtype=float)
        eff = EfficacySpec(style="alpha_spending", family=family)
        fut = FutilitySpec(mode="none")
        spec = BoundaryCalculatorSpec(alpha=alpha, tails=2, efficacy=eff, futility=fut)
        calculator = BoundaryCalculator(spec)
        boundaries = calculator.compute_boundaries(rates)

        payload: Dict[str, object] = {
            str(key): value for key, value in dict(design_payload or {}).items()
        }
        payload.setdefault("info_times", [float(x) for x in rates.tolist()])
        payload.setdefault("planned_max_n", int(planned_max_n))
        payload.setdefault("spending_family", family)
        upper = [float(x) for x in boundaries["upper"]]
        lower_raw = boundaries.get("lower")
        payload["boundaries"] = {
            "upper": upper,
            "lower": [float(x) for x in lower_raw] if lower_raw is not None else None,
        }

        return TwoProportionsProcedure(
            info_times=rates.tolist(),
            z_upper=list(boundaries["upper"]),
            z_lower=(
                list(boundaries["lower"])
                if boundaries.get("lower") is not None
                else None
            ),
            planned_max_n=int(planned_max_n),
            allocation_ratio=float(allocation_ratio),
            design_payload=payload,
        )

    return _factory


@dataclass
class TwoProportionsProcedure:
    """Procedure implementation for the two-proportions simulator."""

    info_times: Sequence[float]
    z_upper: Sequence[float]
    z_lower: Optional[Sequence[float]]
    planned_max_n: int
    allocation_ratio: float = 1.0
    pooled: bool = True
    design_payload: Optional[Mapping[str, object]] = None

    _cum_nA: int = field(default=0, init=False)
    _cum_mA: int = field(default=0, init=False)
    _cum_nB: int = field(default=0, init=False)
    _cum_mB: int = field(default=0, init=False)
    _last_checked_idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._info_times = np.asarray(list(map(float, self.info_times)), dtype=float)
        if self._info_times.size == 0 or not np.isclose(self._info_times[-1], 1.0):
            raise ValueError("info_times must be non-empty and end at 1.0")
        self._z_upper = [float(x) for x in self.z_upper]
        self._z_lower = (
            [float(x) for x in self.z_lower] if self.z_lower is not None else None
        )
        self._planned_max_n = int(self.planned_max_n)
        self._sample_n_total = self._compute_sample_sizes()
        self._metadata_snapshot = self._build_metadata_snapshot()

    def _compute_sample_sizes(self) -> np.ndarray:
        schedule = simulation.compute_cumulative_sample_sizes(
            [float(x) for x in self.info_times], self._planned_max_n
        )
        return np.asarray(schedule, dtype=int)

    def _build_metadata_snapshot(self) -> Dict[str, object]:
        raw_payload = dict(self.design_payload or {})
        payload: Dict[str, object] = {
            str(key): value for key, value in raw_payload.items()
        }
        sample_sizes = [int(x) for x in self._sample_n_total.tolist()]
        payload.setdefault("info_times", [float(x) for x in self._info_times])
        payload.setdefault("planned_max_n", int(self._planned_max_n))
        payload.setdefault("sample_sizes", sample_sizes)
        payload.setdefault(
            "boundaries",
            {
                "upper": [float(x) for x in self._z_upper],
                "lower": (
                    [float(x) for x in self._z_lower]
                    if self._z_lower is not None
                    else None
                ),
            },
        )
        payload.setdefault("allocation_ratio", float(self.allocation_ratio))
        payload.setdefault("n_looks", len(sample_sizes))
        return payload

    def ingest(self, cumulative: Mapping[str, object]) -> None:
        self._cum_nA += _coerce_int(cumulative.get("nA", 0))
        self._cum_mA += _coerce_int(cumulative.get("mA", 0))
        self._cum_nB += _coerce_int(cumulative.get("nB", 0))
        self._cum_mB += _coerce_int(cumulative.get("mB", 0))

    def should_stop(self, look: int) -> Optional[Dict[str, object]]:
        total = int(self._cum_nA + self._cum_nB)
        idx = 0
        for analysis_idx, required in enumerate(self._sample_n_total, start=1):
            if total >= int(required):
                idx = analysis_idx
        if idx <= self._last_checked_idx:
            return None

        for analysis in range(self._last_checked_idx + 1, idx + 1):
            z = compute_wald_z(
                nA=self._cum_nA,
                mA=self._cum_mA,
                nB=self._cum_nB,
                mB=self._cum_mB,
                pooled=self.pooled,
            )
            upper = float(self._z_upper[analysis - 1])
            lower = (
                float(self._z_lower[analysis - 1])
                if self._z_lower is not None and analysis - 1 < len(self._z_lower)
                else None
            )
            self._last_checked_idx = analysis
            if z >= upper:
                return {
                    "reject": True,
                    "reason": "efficacy",
                    "analysis": analysis,
                    "z": float(z),
                }
            if lower is not None and z <= lower:
                return {
                    "reject": False,
                    "reason": "futility",
                    "analysis": analysis,
                    "z": float(z),
                }
        return None

    def reset(self) -> None:
        self._cum_nA = 0
        self._cum_mA = 0
        self._cum_nB = 0
        self._cum_mB = 0
        self._last_checked_idx = 0

    def snapshot_metadata(self) -> Dict[str, object]:
        return dict(self._metadata_snapshot)


__all__ = [
    "ProcedureFactory",
    "TwoProportionsProcedure",
    "build_two_prop_procedure_factory",
]


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float, np.integer)):
        return int(value)
    if isinstance(value, SupportsInt):
        return int(value)
    raise TypeError(f"Cannot convert {value!r} to int")
