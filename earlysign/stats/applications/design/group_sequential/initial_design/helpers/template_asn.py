"""
Monte-Carlo-backed ASN adapter for template procedures.

This module provides :class:`TemplateASNAdapter`, a thin wrapper that exposes
the :class:`~earlysign.stats.essentials.methods.group_sequential.asn.ASNCalculator`
interface while delegating expected-sample-size evaluation to the simulator
layer. It allows the timing optimiser to work with ledger-backed templates.

Examples
--------
>>> from earlysign.stats.applications.design.group_sequential.initial_design.helpers.template_asn import TemplateASNAdapter
>>> from earlysign.stats.applications.design.group_sequential.initial_design.scenarios.fst_to_gst import _build_two_prop_procedure_factory
>>> from earlysign.stats.essentials.methods.group_sequential.spending import OBFSpending
>>> from earlysign.stats.essentials.schemes.two_proportions.asn import build_asn_calculator
>>> spending = OBFSpending(alpha=0.05, sided=2)
>>> calc = build_asn_calculator(
...     alpha=0.05,
...     beta=0.2,
...     sided=2,
...     p_control=0.5,
...     effect_size=0.1,
...     allocation_ratio=1.0,
...     spending=spending,
... )
>>> proc_factory = _build_two_prop_procedure_factory(
...     spending_obj=spending,
...     alpha=0.05,
...     allocation_ratio=1.0,
... )
>>> adapter = TemplateASNAdapter(
...     procedure_factory=proc_factory,
...     base_calculator=calc,
...     planned_max_n=200,
...     p_control=0.5,
...     n_sim=20,
...     batch_size=None,
...     seed=1234,
... )
>>> adapter.evaluate([0.5, 1.0])
195.0
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from earlysign.stats.applications.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
)
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.schemes.two_proportions.simulator import (
    TwoProportionsSimulator,
    compute_cumulative_sample_sizes,
)

DesignPayloadBuilder = Callable[[Sequence[float], int], Mapping[str, Any]]


class TemplateASNAdapter(ASNCalculator):
    """Expose the ASN-calculator surface for template-backed procedures."""

    def __init__(
        self,
        *,
        procedure_factory: ProcedureFactory,
        base_calculator: ASNCalculator,
        planned_max_n: int,
        p_control: float,
        n_sim: int = 200,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        design_payload_builder: Optional[DesignPayloadBuilder] = None,
    ) -> None:
        self._procedure_factory = procedure_factory
        self._calculator = base_calculator
        self._planned_max_n = int(planned_max_n)
        self._p_control = float(p_control)
        self._n_sim = int(n_sim)
        self._seed = seed
        self._design_payload_builder = design_payload_builder
        self._batch_size = None if batch_size is None else int(batch_size)

        self._simulator = TwoProportionsSimulator(
            effect_size=float(base_calculator.alternative),
            n_simulations=int(n_sim),
            batch_size=self._batch_size,
            allocation_ratio=float(base_calculator.allocation_ratio),
        )

        self.st_dev = float(base_calculator.st_dev)
        self.allocation_ratio = float(base_calculator.allocation_ratio)
        self.alternative = float(base_calculator.alternative)

    def _build_design_payload(
        self, info_times: Sequence[float], planned_max_n: int
    ) -> Dict[str, Any]:
        if self._design_payload_builder is None:
            raw_payload: Mapping[str, Any] = {
                "info_times": list(map(float, info_times)),
                "planned_max_n": int(planned_max_n),
            }
        else:
            raw_payload = self._design_payload_builder(info_times, planned_max_n)
        return {str(key): value for key, value in dict(raw_payload).items()}

    def evaluate(self, info: Sequence[float]) -> float:
        rates = self._validate_information_rates(info)
        payload = self._build_design_payload(rates, self._planned_max_n)
        procedure = self._procedure_factory(
            rates, self._planned_max_n, payload, self._seed
        )
        procedure.reset()
        sample_sizes = compute_cumulative_sample_sizes(rates, self._planned_max_n)
        schedule_sizes = sample_sizes if self._batch_size is None else None
        result = self._simulator.simulate(
            procedure,
            p_control=self._p_control,
            effect_size=float(self.alternative),
            n_simulations=self._n_sim,
            rng_seed=self._seed,
            max_total=self._planned_max_n,
            info_times=rates if self._batch_size is None else None,
            cumulative_sizes=schedule_sizes,
        )
        return float(result.expected_sample_size)

    def _validate_information_rates(self, info: Sequence[float]) -> List[float]:
        validated = np.asarray(
            self._calculator._validate_information_rates(info), dtype=float
        )
        return [float(x) for x in validated.tolist()]

    def _per_stage_alpha(self, rates: Sequence[float]) -> Any:
        return self._calculator._per_stage_alpha(np.asarray(list(rates), dtype=float))

    def _z_boundaries(self, per_stage_alpha: Sequence[float]) -> Any:
        return self._calculator._z_boundaries(
            np.asarray(list(per_stage_alpha), dtype=float)
        )

    def _n_max_from_power(self) -> float:
        return float(self._planned_max_n)
