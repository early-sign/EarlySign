"""Binomial GST design creation logic."""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Type, Union, cast

from earlysign.methods.group_sequential.asn import ASNCalculator
from earlysign.methods.group_sequential.design.initial_design.scenarios.fst_to_gst import (
    AddInterimToFixedSampleTest,
)
from earlysign.methods.group_sequential.spending import (
    SpendingFunction,
    get_spending_class,
)
from earlysign.stats.schemes.two_proportions.asn import (
    build_asn_calculator,
)
from earlysign.stats.schemes.two_proportions.design import (
    build_two_proportions_scheme,
)


def create_binomial_design(
    *,
    alpha: float,
    delta: float,
    power: float,
    p_control: float,
    allocation_ratio: float = 1.0,
    spending: Union[SpendingFunction, Type[SpendingFunction], str] = "pocock",
    effect_sizes: Optional[Sequence[float]] = None,
    n_sim: int = 200,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    design_payload_builder: Optional[
        Callable[[Sequence[float], int], Mapping[str, Any]]
    ] = None,
) -> AddInterimToFixedSampleTest:
    """Return a configured binomial GST design helper.

    This surfaces the class-based ``AddInterimToFixedSampleTest`` flow
    through a standalone API.
    """

    if hasattr(spending, "cumulative") and hasattr(
        spending, "boundaries_from_stage_alpha"
    ):
        spending_obj = cast(SpendingFunction, spending)
    else:
        spending_cls = (
            cast(Type[SpendingFunction], spending)
            if isinstance(spending, type)
            else get_spending_class(str(spending))
        )
        spending_obj = spending_cls(alpha=alpha)

    base_scheme = build_two_proportions_scheme(
        p_control=p_control,
        target_effect=delta,
        effect_sizes=effect_sizes or [delta],
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
    )
    resolved_scheme = base_scheme.with_effect_sizes(effect_sizes)

    def _asn_factory() -> ASNCalculator:
        return build_asn_calculator(
            alpha=alpha,
            beta=1.0 - power,
            sided=2,
            p_control=p_control,
            effect_size=delta,
            allocation_ratio=allocation_ratio,
            spending=spending_obj,
        )

    procedure_factory = resolved_scheme.procedure_factory_builder(
        spending_obj, allocation_ratio
    )

    return AddInterimToFixedSampleTest(
        alpha=alpha,
        power=power,
        scheme=resolved_scheme,
        procedure_factory=procedure_factory,
        asn_calculator_factory=_asn_factory,
        simulator=resolved_scheme.simulator_factory(
            int(n_sim), float(allocation_ratio)
        ),
        batch_size=batch_size,
        seed=seed,
        design_payload_builder=design_payload_builder,
    )
