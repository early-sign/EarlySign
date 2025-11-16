"""
Procedure builders for two-proportions group-sequential simulations.

These helpers produce ``Procedure`` instances compatible with the simulation
utilities under ``earlysign.stats.essentials`` without relying on higher-level
application modules.

Examples
--------
>>> from earlysign.stats.essentials.methods.group_sequential import simulation
>>> from earlysign.stats.essentials.methods.group_sequential.spending import OBFSpending
>>> from earlysign.stats.essentials.schemes.two_proportions.procedure import (
...     TwoProportionsBatchedProcedure,
... )
>>> from earlysign.stats.essentials.schemes.two_proportions.simulator import (
...     TwoProportionsSimulationRequest,
...     TwoProportionsSimulator,
... )
>>> spending = OBFSpending(alpha=0.05, sided=2)
>>> factory = TwoProportionsBatchedProcedure.factory_builder(
...     spending_obj=spending,
...     alpha=0.05,
...     allocation_ratio=1.0,
... )
>>> procedure = factory(
...     info_times=[0.5, 1.0],
...     planned_max_n=200,
...     design_payload=None,
...     rng_seed=0,
... )
>>> sampling = simulation.InfoTimeSampling(
...     info_times=[0.5, 1.0],
...     planned_max_n=200,
...     allocation_ratio=1.0,
... )
>>> simulator = TwoProportionsSimulator(
...     effect_size=0.1,
...     n_simulations=30,
...     allocation_ratio=1.0,
...     strategy=None,
... )
>>> request = TwoProportionsSimulationRequest(
...     p_control=0.5,
...     effect_size=0.1,
...     n_simulations=30,
...     max_total=200,
...     sampling=sampling,
... )
>>> point = simulator.simulate(procedure, requests=[request], rng_seed=0)[0]
>>> round(point.expected_sample_size, 1)
190.0
>>> round(point.power, 2)
0.3
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)
from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    BatchedProcedure,
)
from earlysign.stats.essentials.methods.group_sequential.spending import (
    SpendingFunction,
)
from earlysign.stats.essentials.schemes.two_proportions.wald_z import (
    compute_wald_z,
    compute_wald_z_array,
)

ProcedureFactory = Callable[
    [Sequence[float], int, Optional[Mapping[str, object]], Optional[int]],
    "TwoProportionsProcedure",
]
TwoProportionsBatchedProcedureFactory = Callable[
    [Sequence[float], int, Optional[Mapping[str, object]], Optional[int]],
    "TwoProportionsBatchedProcedure",
]


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

    @classmethod
    def factory_builder(
        cls,
        *,
        spending_obj: SpendingFunction,
        alpha: float,
        allocation_ratio: float,
    ) -> ProcedureFactory:
        """Return a factory that builds :class:`TwoProportionsProcedure`."""

        family = spending_obj.name

        def _factory(
            info_times: Sequence[float],
            planned_max_n: int,
            design_payload: Optional[Mapping[str, object]],
            rng_seed: Optional[int],
        ) -> "TwoProportionsProcedure":
            rates = np.asarray(info_times, dtype=float)
            eff = EfficacySpec(style="alpha_spending", family=family)
            fut = FutilitySpec(mode="none")
            spec = BoundaryCalculatorSpec(
                alpha=alpha,
                tails=2,
                efficacy=eff,
                futility=fut,
            )
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
                "lower": (
                    [float(x) for x in lower_raw] if lower_raw is not None else None
                ),
            }

            return cls(
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

    def ingest(self, cumulative: Mapping[str, Any]) -> None:
        self._cum_nA += int(cumulative.get("nA", 0))
        self._cum_mA += int(cumulative.get("mA", 0))
        self._cum_nB += int(cumulative.get("nB", 0))
        self._cum_mB += int(cumulative.get("mB", 0))

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


class TwoProportionsBatchedProcedure(BatchedProcedure):
    """Vectorized variant of :class:`TwoProportionsProcedure`.

    This adapter exposes the ``BatchedProcedure`` protocol so simulators can
    evaluate many simulations at once using NumPy arrays while still
    supporting the scalar interface for compatibility.
    """

    def __init__(self, base: TwoProportionsProcedure):
        self._base = base
        self._z_upper = [float(x) for x in base.z_upper]
        self._z_lower = (
            [float(x) for x in base.z_lower] if base.z_lower is not None else None
        )
        self.pooled = bool(base.pooled)
        self.allocation_ratio = float(base.allocation_ratio)
        self._planned_max_n = int(base.planned_max_n)
        self._batch_cum_nA = np.array([], dtype=int)
        self._batch_cum_mA = np.array([], dtype=int)
        self._batch_cum_nB = np.array([], dtype=int)
        self._batch_cum_mB = np.array([], dtype=int)

    @classmethod
    def factory_builder(
        cls,
        *,
        spending_obj: SpendingFunction,
        alpha: float,
        allocation_ratio: float,
    ) -> TwoProportionsBatchedProcedureFactory:
        """Return a factory that builds :class:`TwoProportionsBatchedProcedure`."""

        base_factory = TwoProportionsProcedure.factory_builder(
            spending_obj=spending_obj,
            alpha=alpha,
            allocation_ratio=allocation_ratio,
        )

        def _factory(
            info_times: Sequence[float],
            planned_max_n: int,
            design_payload: Optional[Mapping[str, object]],
            rng_seed: Optional[int],
        ) -> "TwoProportionsBatchedProcedure":
            base = base_factory(info_times, planned_max_n, design_payload, rng_seed)
            return cls.from_procedure(base)

        return _factory

    @classmethod
    def from_procedure(
        cls, base: TwoProportionsProcedure
    ) -> "TwoProportionsBatchedProcedure":
        return cls(base)

    # Scalar Procedure API -------------------------------------------------
    def ingest(self, cumulative: Mapping[str, object]) -> None:
        self._base.ingest(cumulative)

    def should_stop(self, look: int) -> Optional[Dict[str, object]]:
        return self._base.should_stop(look)

    def reset(self) -> None:
        self._base.reset()

    def snapshot_metadata(self) -> Dict[str, object]:
        return self._base.snapshot_metadata()

    # BatchedProcedure API -------------------------------------------------
    def reset_batch(self, n_simulations: int) -> None:
        size = int(n_simulations)
        self._batch_cum_nA = np.zeros(size, dtype=int)
        self._batch_cum_mA = np.zeros(size, dtype=int)
        self._batch_cum_nB = np.zeros(size, dtype=int)
        self._batch_cum_mB = np.zeros(size, dtype=int)

    def ingest_batch(
        self,
        cumulative: Dict[str, np.ndarray],
        *,
        active_mask: np.ndarray,
    ) -> None:
        mask = np.asarray(active_mask, dtype=bool)
        if self._batch_cum_nA.size == 0:
            raise RuntimeError("Call reset_batch() before ingest_batch().")
        nA = self._coerce_batch_array(cumulative.get("nA", 0))
        mA = self._coerce_batch_array(cumulative.get("mA", 0))
        nB = self._coerce_batch_array(cumulative.get("nB", 0))
        mB = self._coerce_batch_array(cumulative.get("mB", 0))
        self._batch_cum_nA[mask] += nA[mask]
        self._batch_cum_mA[mask] += mA[mask]
        self._batch_cum_nB[mask] += nB[mask]
        self._batch_cum_mB[mask] += mB[mask]

    def should_stop_batch(
        self,
        look: int,
        *,
        active_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(active_mask, dtype=bool)
        stop_mask = np.zeros_like(mask, dtype=bool)
        reject_mask = np.zeros_like(mask, dtype=bool)
        if not np.any(mask):
            return stop_mask, reject_mask

        z_vals = compute_wald_z_array(
            self._batch_cum_nA,
            self._batch_cum_mA,
            self._batch_cum_nB,
            self._batch_cum_mB,
            pooled=self.pooled,
        )
        upper = float(self._z_upper[look - 1])
        hits_upper = mask & (z_vals >= upper)
        stop_mask |= hits_upper
        reject_mask |= hits_upper
        lower = (
            float(self._z_lower[look - 1])
            if self._z_lower is not None and look - 1 < len(self._z_lower)
            else None
        )
        if lower is not None:
            hits_lower = mask & (z_vals <= lower)
            stop_mask |= hits_lower

        return stop_mask, reject_mask

    # Internal utilities ---------------------------------------------------
    def _coerce_batch_array(self, values: object) -> np.ndarray:
        arr = np.asarray(values, dtype=int)
        if arr.ndim == 0:
            arr = np.full(self._batch_cum_nA.shape[0], int(arr), dtype=int)
        if arr.shape[0] != self._batch_cum_nA.shape[0]:
            raise ValueError("Batch array length mismatch")
        return arr
