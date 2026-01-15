"""UI-oriented enums for initial design helpers."""

from enum import Enum


class DesignMode(str, Enum):
    """Design optimization modes supported by the notebook UI."""

    NMAX_FIXED_MIN_MDE = "nmax_fixed_min_mde"
    FIXED_TIMING = "fixed_timing"
    OPTIMIZE_ASN = "optimize_asn"
    OPTIMIZE_DESIGN = "optimize_design"
    FIXED_POWER = "fixed_power"
