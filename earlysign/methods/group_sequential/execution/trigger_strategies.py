from typing import Any, List, Optional, Tuple

import numpy as np

from earlysign.framework.trace import Traced, extract_traces
from earlysign.schema.ES3.GST import (
    DueLookTrigger,
    EquidistantSchedule,
    EventCountTimer,
    FisherInformationTimer,
    FixedSchedule,
    Protocol,
    SampleSizeTimer,
)
from earlysign.schema.ES3.GST.Log import LookResult, ScheduleTrigger


def get_pending_look_trigger(
    protocol: Traced[Protocol],
    metrics: Traced[Any],
    history: Traced[List[Tuple[int, LookResult]]],
) -> Optional[Traced[ScheduleTrigger]]:
    """
    Identifies if a planned look is "due" and returns a ScheduleTrigger if so.

    A planned look i is due if:
    1. current_info_frac >= planned_info_frac[i] - tolerance
    2. No LookResult with that look index exists in the history trajectory.

    Args:
        protocol: The GST Protocol definition (Traced).
        metrics: The current Scoreboard (containing accrued sample size/events) (Traced).
        history: The trajectory of previous Looks recorded in the ledger (Traced).

    Returns:
        ScheduleTrigger if a look is due, else None. (Traced)
    """
    policy = protocol.data.method.stopping_policy
    trigger_spec = policy.trigger_strategy
    trace = extract_traces(protocol, metrics, history)

    if not isinstance(trigger_spec, DueLookTrigger):
        return None

    tolerance = trigger_spec.tolerance if trigger_spec.tolerance is not None else 0.0
    schedule = policy.schedule
    timer = policy.timer

    # 1. Determine planned points (informational times t_1, ..., t_K)
    if isinstance(schedule, FixedSchedule):
        planned_points = schedule.analyses
    elif isinstance(schedule, EquidistantSchedule):
        planned_points = np.linspace(
            1.0 / schedule.n_looks, 1.0, schedule.n_looks
        ).tolist()
    else:
        return None

    # 2. Calculate current information fraction
    n_max: float = 0.0
    if isinstance(timer, SampleSizeTimer):
        n_max = float(sum(timer.max_sample_size.values()))
    elif isinstance(timer, EventCountTimer):
        n_max = timer.max_events
    elif isinstance(timer, FisherInformationTimer):
        n_max = timer.max_information
    else:
        return None

    total_n = sum(arm.metrics.total for arm in metrics.data.arms.values())
    current_info_frac = total_n / n_max if n_max > 0 else 0.0

    # 3. Identify due looks
    due_indices = []
    for i, t_plan in enumerate(planned_points):
        look_num = i + 1
        # Check if current information exceeds the planned threshold (minus tolerance)
        if current_info_frac >= t_plan - tolerance:
            # Check if this planned look index has already been recorded in the history
            is_recorded = any(state.look == look_num for _, state in history.data)
            if not is_recorded:
                due_indices.append(look_num)

    if not due_indices:
        return None

    # 4. Return trigger for the highest due index (skipping intermediate looks if data comes fast)
    highest_look = max(due_indices)
    trigger = ScheduleTrigger(
        kind="schedule", index=highest_look, value=planned_points[highest_look - 1]
    )
    return Traced(data=trigger, trace=trace or [])
