"""Boundary records for group sequential analyses."""

from earlysign.framework.records import LedgerRecord, QueryMixin


class GroupSequentialBoundaryRecord(LedgerRecord, QueryMixin):
    """
    Nominal boundaries resolved from a GS design.

    Stores efficacy and futility boundaries resolved at a given
    information time, plus design metadata.

    Payload example:
      {
        "upper": 2.5,
        "lower": -2.5,
        "efficacy": {"upper": 2.5},
        "futility": {"lower": -2.5, "binding": false, "mode": "none"},
        "scale": "z",
        "statistic_type": "wald_z",
        "statistic_scale": "z",
        "alpha": 0.05,
        "tails": 2,
        "info_time": 0.5,
        "look": 2
      }
    """

    schema = {
        "upper": float,
        "lower": float,
        "efficacy": (dict | None, None),
        "futility": (dict | None, None),
        "scale": str,
        "statistic_type": (str | None, None),
        "statistic_scale": (str | None, None),
        "alpha": (float | None, None),
        "tails": (int | None, None),
        "info_time": (float | None, None),
        "look": (int | None, None),
    }
