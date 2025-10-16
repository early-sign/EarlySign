"""Type definitions for group sequential trial design."""

from enum import Enum


class SpendingFunction(str, Enum):
    """Alpha spending function types.

    >>> SpendingFunction.OBRIEN_FLEMING.value
    'obrien_fleming'
    >>> SpendingFunction.POCOCK.value
    'pocock'
    """

    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    HSD = "hsd"


class TestType(str, Enum):
    """Statistical test types supported.

    >>> TestType.TWO_PROPORTIONS.value
    'two_sample_proportions'
    """

    TWO_PROPORTIONS = "two_sample_proportions"
    TWO_MEANS = "two_sample_means"
    TIME_TO_EVENT = "time_to_event"


class TimingType(str, Enum):
    """Timing specification for interim analyses.

    >>> TimingType.INFORMATION_BASED.value
    'information_based'
    """

    INFORMATION_BASED = "information_based"
    CALENDAR_TIME = "calendar_time"


class InformationSpacing(str, Enum):
    """Information time spacing strategies.

    >>> InformationSpacing.EQUAL.value
    'equal'
    """

    EQUAL = "equal"
    CUSTOM = "custom"


class DesignMode(str, Enum):
    """Design optimization modes.

    >>> DesignMode.FIXED_TIMING.value
    'fixed_timing'
    """

    NMAX_FIXED_MIN_MDE = "nmax_fixed_min_mde"
    FIXED_TIMING = "fixed_timing"
    OPTIMIZE_ASN = "optimize_asn"
    OPTIMIZE_DESIGN = "optimize_design"
    FIXED_POWER = "fixed_power"
