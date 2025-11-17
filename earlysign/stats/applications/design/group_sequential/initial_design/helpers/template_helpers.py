"""
Helper utilities for integrating ledger-backed templates with the
group-sequential initial-design workflow.

The primary entry point is :func:`build_template_procedure_factory` which
turns a template factory into a :class:`ProcedureFactory`.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence

from earlysign.stats.applications.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
    ProcedureLike,
)

if TYPE_CHECKING:
    from earlysign.framework.templates import TemplateBase


def build_template_procedure_factory(
    *,
    template_factory: Callable[[Any, str, Optional[str]], "TemplateBase"],
    stop_decision_fn: Callable[["TemplateBase", int], Optional[Dict[str, Any]]],
    experiment_id: str,
    table_name: Optional[str] = None,
) -> ProcedureFactory:
    """Return a :class:`ProcedureFactory` that wraps ``TemplateProcedureAdapter``."""

    def _factory(
        _info_times: Sequence[float],
        _planned_max_n: int,
        design_payload: Optional[Mapping[str, Any]],
        rng_seed: Optional[int],
    ) -> ProcedureLike:
        from earlysign.stats.applications.design.group_sequential.initial_design.scenarios.fst_to_gst import (
            TemplateProcedureAdapter,
        )

        return TemplateProcedureAdapter(
            template_factory=template_factory,
            experiment_id=experiment_id,
            table_name=table_name,
            stop_decision_fn=stop_decision_fn,
            design_payload=dict(design_payload or {}),
            rng_seed=rng_seed,
        )

    return _factory
