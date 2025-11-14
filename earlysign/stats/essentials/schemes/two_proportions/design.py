"""
Scheme adaptor for two-proportions group-sequential designs.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, cast

from earlysign.stats.applications.design.group_sequential.initial_design.helpers.scheme import (
    GSTSchemeHooks,
)
from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    Simulator,
)
from earlysign.stats.essentials.methods.group_sequential.spending import (
    SpendingFunction,
)
from earlysign.stats.essentials.schemes.two_proportions.asn import (
    build_asn_calculator,
)
from earlysign.stats.essentials.schemes.two_proportions.effect_size import (
    TwoProportionsEffectSizeCalculator,
)
from earlysign.stats.essentials.schemes.two_proportions.procedure import (
    build_two_prop_procedure_factory,
)
from earlysign.stats.essentials.schemes.two_proportions.simulator import (
    TwoProportionsSimulator,
)


def build_two_proportions_scheme(
    *,
    p_control: float,
    target_effect: float,
    effect_sizes: Sequence[float],
    alpha: float,
    power: float,
    allocation_ratio: float,
) -> GSTSchemeHooks:
    """Return a :class:`GSTSchemeHooks` configured for two-proportion Z tests."""

    effect_grid = [float(x) for x in effect_sizes]

    def fsd_planner() -> Dict[str, int]:
        calc = TwoProportionsEffectSizeCalculator(p_control=float(p_control))
        n_per_group = calc.calculate_sample_size(
            effect_size=float(target_effect), alpha=float(alpha), power=float(power)
        )
        planned_total = int(round((1.0 + float(allocation_ratio)) * n_per_group))
        return {
            "n_fsd_per_group": int(n_per_group),
            "planned_max_n": int(planned_total),
        }

    def simulator_factory(n_sim: int, alloc_ratio: float) -> Simulator:
        return cast(
            Simulator,
            TwoProportionsSimulator(
                effect_size=float(target_effect),
                n_simulations=int(n_sim),
                allocation_ratio=float(alloc_ratio),
                strategy=None,
            ),
        )

    def simulator_kwargs_builder(
        effect_size: float,
        planned_max_n: int,
        sampling_strategy: simulation.SamplingStrategy,
    ) -> Dict[str, object]:
        return {
            "p_control": float(p_control),
            "effect_size": float(effect_size),
            "max_total": int(planned_max_n),
            "sampling": sampling_strategy,
        }

    def asn_factory_builder(
        spending_obj: SpendingFunction,
    ) -> Callable[[], ASNCalculator]:
        def _factory() -> ASNCalculator:
            return build_asn_calculator(
                alpha=float(alpha),
                beta=1.0 - float(power),
                sided=2,
                p_control=float(p_control),
                effect_size=float(target_effect),
                allocation_ratio=float(allocation_ratio),
                spending=spending_obj,
            )

        return _factory

    return GSTSchemeHooks(
        name="two_proportions",
        target_effect=float(target_effect),
        effect_sizes=effect_grid,
        null_reference=float(p_control),
        fsd_planner=fsd_planner,
        simulator_factory=simulator_factory,
        simulator_kwargs_builder=simulator_kwargs_builder,
        asn_factory_builder=asn_factory_builder,
        procedure_factory_builder=lambda s, alloc: build_two_prop_procedure_factory(
            spending_obj=s,
            alpha=float(alpha),
            allocation_ratio=float(alloc),
        ),
    )
