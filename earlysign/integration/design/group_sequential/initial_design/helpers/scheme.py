"""
Scheme hooks shared by group-sequential design helpers.

The adapters defined here capture scheme-specific callbacks (e.g.,
fixed-sample planners and simulator builders) so scenario code can remain
agnostic to the underlying statistical family.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from earlysign.integration.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
)
from earlysign.stats.methods.group_sequential import simulation
from earlysign.stats.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.methods.group_sequential.operating_characteristics import (
    Simulator,
)
from earlysign.stats.methods.group_sequential.spending import SpendingFunction

FSDPlanner = Callable[[], Dict[str, int]]
SimulatorFactory = Callable[[int, float], Simulator]
SimulationRequestBuilder = Callable[
    [float, int, simulation.SamplingStrategy, int],
    Any,
]
ASNFactoryBuilder = Callable[[SpendingFunction], Callable[[], ASNCalculator]]
ProcedureFactoryBuilder = Callable[[SpendingFunction, float], ProcedureFactory]
SamplingStrategyBuilder = Callable[
    [Sequence[float], int, Optional[int]],
    simulation.SamplingStrategy,
]


@dataclass(frozen=True)
class GSTSchemeHooks:
    """Scheme-specific hooks consumed by :class:`AddInterimToFixedSampleTest`."""

    name: str
    target_effect: float
    effect_sizes: Sequence[float]
    null_reference: Optional[float]
    fsd_planner: FSDPlanner
    simulator_factory: SimulatorFactory
    simulation_request_builder: SimulationRequestBuilder
    sampling_strategy_builder: SamplingStrategyBuilder
    asn_factory_builder: ASNFactoryBuilder
    procedure_factory_builder: ProcedureFactoryBuilder

    def with_effect_sizes(
        self, overrides: Optional[Sequence[float]]
    ) -> "GSTSchemeHooks":
        """Return a copy with alternate effect sizes if provided."""
        if overrides is None:
            return self
        return GSTSchemeHooks(
            name=self.name,
            target_effect=self.target_effect,
            effect_sizes=[float(x) for x in overrides],
            null_reference=self.null_reference,
            fsd_planner=self.fsd_planner,
            simulator_factory=self.simulator_factory,
            simulation_request_builder=self.simulation_request_builder,
            sampling_strategy_builder=self.sampling_strategy_builder,
            asn_factory_builder=self.asn_factory_builder,
            procedure_factory_builder=self.procedure_factory_builder,
        )
