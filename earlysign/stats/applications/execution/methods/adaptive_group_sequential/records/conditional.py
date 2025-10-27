"""Adaptive design support records for group sequential workflows."""

from earlysign.framework.records import LedgerRecord, QueryMixin


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
