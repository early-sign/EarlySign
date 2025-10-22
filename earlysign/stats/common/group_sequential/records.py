"""
Core records for scheme-agnostic Group Sequential components.

These records are intentionally minimal and free of scheme-specific payloads.
Operators in this package (e.g., info_time, design, boundary, decision) read
and write these records.
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class InformationTimeRecord(LedgerRecord, QueryMixin):
    """
    Information time snapshots (scheme-agnostic).
    Stores information time t in [0, 1].

    Payload example:
      {"info_time": 0.5}
    """

    schema = {
        "info_time": float,
    }


class GroupSequentialDesignRecord(LedgerRecord, QueryMixin):
    """
    Design record for Group Sequential Testing.

    Stores the complete design specification including spending functions,
    boundaries, and other configuration.

    Payload example:
      {
        "alpha": 0.05,
        "tails": 2,
        "scale": "z",
        "efficacy": {"style": "alpha_spending", "family": "obf"},
        "futility": {"mode": "none"},
        "binding_mode": "non_binding",
        "planned_max_n": 1000,
        "planned_info_times": [0.33, 0.67, 1.0]
      }
    """

    schema = {
        "alpha": float,
        "tails": int,
        "scale": str,
        "efficacy": dict,
        "futility": dict,
        "binding_mode": (str | None, "non_binding"),
        "planned_max_n": (int | None, None),
        "planned_info_times": (list, [1.0]),
    }


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
        "alpha": (float | None, None),
        "tails": (int | None, None),
        "info_time": (float | None, None),
        "look": (int | None, None),
    }


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
        "info_time": (float | None, None),
    }


class ConditionalPowerRecord(LedgerRecord, QueryMixin):
    """
    Conditional power calculation result for adaptive designs.

    Stores conditional power computed at an interim analysis, given
    observed results and assumed future effect size.

    Payload example:
      {
        "conditional_power": 0.85,
        "observed_z": 2.1,
        "current_info_time": 0.5,
        "final_info_time": 1.0,
        "assumed_effect": 0.5
      }
    """

    schema = {
        "conditional_power": float,
        "observed_z": float,
        "current_info_time": float,
        "final_info_time": float,
        "assumed_effect": float,
    }


class DesignUpdateDecisionRecord(LedgerRecord, QueryMixin):
    """
    Adaptive design update decision based on conditional power.

    Stores decision whether to continue, stop, or modify trial design
    based on promising-zone logic and conditional power threshold.

    Payload example:
      {
        "decision": "continue" | "stop_futility" | "stop_efficacy",
        "conditional_power": 0.85,
        "in_promising_zone": true,
        "recommendation": "continue with original design",
        "observed_z": 2.1,
        "current_info_time": 0.5
      }
    """

    schema = {
        "decision": str,
        "conditional_power": float,
        "in_promising_zone": bool,
        "recommendation": str,
        "observed_z": float,
        "current_info_time": float,
    }


class UpdatedBoundariesRecord(LedgerRecord, QueryMixin):
    """
    Updated boundaries after adaptive interim modification.

    Stores recomputed boundaries for remaining analyses after
    trial design modification to maintain Type I error control.

    Payload example:
      {
        "remaining_info_times": [0.75, 1.0],
        "upper_bounds": [2.2, 1.96],
        "lower_bounds": [-0.5, -0.5],
        "alpha_remaining": 0.020,
        "alpha_increments": [0.005, 0.015]
      }
    """

    schema = {
        "remaining_info_times": list,
        "upper_bounds": list,
        "lower_bounds": list,
        "alpha_remaining": float,
        "alpha_increments": list,
    }
