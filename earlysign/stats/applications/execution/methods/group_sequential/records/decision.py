"""Decision records for group sequential analyses."""

from earlysign.framework.records import LedgerRecord, QueryMixin


class GroupSequentialDecisionSignalRecord(LedgerRecord, QueryMixin):
    """
    Stop/continue signal produced by comparing a statistic to boundaries.

    Stores a decision emitted by comparing a statistic (e.g., Wald Z)
    against the boundaries.

    Payload example:
      {
        "signal": "stop_efficacy" | "stop_futility" | "continue",
        "reason": "efficacy" | "futility" | "none",
        "value": 2.34,
        "value_scale": "z",
        "upper": 2.5,
        "lower": -2.5,
        "scale": "z",
        "statistic_type": "wald_z",
        "statistic_scale": "z",
        "info_time": 0.5
      }
    """

    schema = {
        "signal": str,
        "reason": str,
        "value": (float | None, None),
        "value_scale": (str | None, None),
        "upper": (float | None, None),
        "lower": (float | None, None),
        "scale": (str | None, None),
        "statistic_type": (str | None, None),
        "statistic_scale": (str | None, None),
        "info_time": (float | None, None),
    }
