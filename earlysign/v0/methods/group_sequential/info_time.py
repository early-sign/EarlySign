"""
Information time calculation functions for group sequential designs.

This module provides pure functions for calculating information time t ∈ [0, 1],
which represents the proportion of planned information accrued at a given point.

Information time is fundamental to group sequential testing:
- Determines where we are on the spending function α(t)
- Used to compute boundaries at each analysis
- Can be based on sample size, events, variance, or Fisher information

Functions
---------
Sample-based:
    info_time_from_sample_size(n_current, n_max) -> float
    info_time_from_samples_by_group(n_control, n_treatment, n_max_control, n_max_treatment) -> float

Ratio-based (general):
    info_time_from_ratio(info_now, info_max) -> float

Variance-based:
    info_time_from_variance(var_now, var_target) -> float
    info_time_from_sd(sd_now, sd_target) -> float

Fisher information-based:
    info_time_from_fisher(fisher_now, fisher_max) -> float

Examples
--------
>>> # Sample-based information time
>>> t = info_time_from_sample_size(n_current=500, n_max=2000)
>>> t
0.25

>>> # Variance-based (information ∝ 1/variance)
>>> t = info_time_from_variance(var_now=0.04, var_target=0.01)
>>> t
0.25
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from earlysign.core.ledger import Ledger
from earlysign.v0.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.v0.framework.records import LedgerRecord, QueryMixin
from earlysign.v0.methods.group_sequential.schemes.two_proportions.binomial_arms import (
    BinomialArmSnapshot,
)

# ============================================================================
# Helper functions
# =============================================================================


def clip01(x: float) -> float:
    """
    Helper function: clip a float into [0, 1].

    Utility function for ensuring information times stay in valid range.

    Parameters
    ----------
    x : float
        Value to clip.

    Returns
    -------
    float
        Value clipped to [0, 1].

    Examples
    --------
    >>> clip01(-0.1)
    0.0
    >>> clip01(0.5)
    0.5
    >>> clip01(1.5)
    1.0
    """
    return max(0.0, min(1.0, float(x)))


# =============================================================================
# Core Information Time Calculations
# =============================================================================


def info_time_from_sample_size(n_current: int, n_max: int) -> float:
    """
    Calculate information time from sample sizes.

    Most common method: t = n_current / n_max

    Parameters
    ----------
    n_current : int
        Current total sample size.
    n_max : int
        Planned maximum total sample size.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> info_time_from_sample_size(n_current=100, n_max=400)
    0.25

    >>> info_time_from_sample_size(n_current=400, n_max=400)
    1.0

    >>> # Over-recruitment is clipped to 1.0
    >>> info_time_from_sample_size(n_current=500, n_max=400)
    1.0
    """
    if n_max <= 0:
        raise ValueError(f"n_max must be positive, got {n_max}")
    if n_current < 0:
        raise ValueError(f"n_current must be non-negative, got {n_current}")

    return clip01(float(n_current) / float(n_max))


def info_time_from_samples_by_group(
    n_control: int, n_treatment: int, n_max_control: int, n_max_treatment: int
) -> float:
    """
        Calculate information time from per-group sample sizes.

        For unequal allocation ratios, information time is based on the
        harmonic mean of the two groups' information times (approximately).

        Parameters
        ----------
        n_control : int
            Current control group sample size.
        n_treatment : int
            Current treatment group sample size.
        n_max_control : int
            Planned maximum control group sample size.
        n_max_treatment : int
            Planned maximum treatment group sample size.

        Returns
        -------
        float
            Information time in [0, 1].

        Examples
        --------
        >>> # Equal allocation, equal accrual
        >>> info_time_from_samples_by_group(50, 50, 200, 200)
        0.25

        >>> # Unequal allocation (1:2 ratio)
        >>> info_time_from_samples_by_group(50, 100, 200, 400)
        0.25

        Notes
        -----
    Information for comparing two groups depends on both sample sizes.
    For proportions with equal allocation:
        I(t) ∝ n_control(t) · n_treatment(t) / (n_control(t) + n_treatment(t))

    This function uses the simpler total sample size approach:
        t = (n_control + n_treatment) / (n_max_control + n_max_treatment)
    """
    n_current = n_control + n_treatment
    n_max = n_max_control + n_max_treatment
    return info_time_from_sample_size(n_current, n_max)


def info_time_from_ratio(info_now: float, info_max: float) -> float:
    """
    Generic information time from any information metric ratio.

    Use when you have a direct measure of information (Fisher information,
    precision, number of events, etc.) and want t = info_now / info_max.

    Parameters
    ----------
    info_now : float
        Current information level.
    info_max : float
        Maximum planned information level.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # Event-based (survival analysis)
    >>> info_time_from_ratio(info_now=75, info_max=300)
    0.25

    >>> # Fisher information
    >>> info_time_from_ratio(info_now=1000.0, info_max=4000.0)
    0.25
    """
    if info_max <= 0:
        raise ValueError(f"info_max must be positive, got {info_max}")
    if info_now < 0:
        raise ValueError(f"info_now must be non-negative, got {info_now}")

    return clip01(float(info_now) / float(info_max))


def info_time_from_variance(var_now: float, var_target: float) -> float:
    """
    Calculate information time from variance (information ∝ 1/variance).

    When variance decreases with sample size, information increases.
    t = var_target / var_now (inverse relationship).

    Parameters
    ----------
    var_now : float
        Current variance estimate.
    var_target : float
        Target (final) variance.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # Variance decreases with sample size
    >>> info_time_from_variance(var_now=0.04, var_target=0.01)
    0.25

    >>> # At target variance, t=1
    >>> info_time_from_variance(var_now=0.01, var_target=0.01)
    1.0

    Notes
    -----
    Variance-based information time is particularly useful when:
    - Sample sizes vary by group or time
    - Adaptive designs adjust allocation ratios
    - Precision is the primary endpoint consideration
    """
    if var_target <= 0:
        raise ValueError(f"var_target must be positive, got {var_target}")
    if var_now <= 0:
        raise ValueError(f"var_now must be positive, got {var_now}")

    return clip01(float(var_target) / float(var_now))


def info_time_from_sd(sd_now: float, sd_target: float) -> float:
    """
    Calculate information time from standard deviation.

    Converts SD to variance and uses info_time_from_variance.
    t = (sd_target / sd_now)²

    Parameters
    ----------
    sd_now : float
        Current standard deviation estimate.
    sd_target : float
        Target (final) standard deviation.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # SD decreases from 0.2 to 0.1 (variance 0.04 → 0.01)
    >>> info_time_from_sd(sd_now=0.2, sd_target=0.1)
    0.25

    >>> info_time_from_sd(sd_now=0.1, sd_target=0.1)
    1.0
    """
    if sd_target <= 0:
        raise ValueError(f"sd_target must be positive, got {sd_target}")
    if sd_now <= 0:
        raise ValueError(f"sd_now must be positive, got {sd_now}")

    var_now = sd_now**2
    var_target = sd_target**2
    return info_time_from_variance(var_now, var_target)


def info_time_from_fisher(fisher_now: float, fisher_max: float) -> float:
    """
    Calculate information time from Fisher information.

    Fisher information is a direct measure of statistical information.
    t = fisher_now / fisher_max

    Parameters
    ----------
    fisher_now : float
        Current Fisher information.
    fisher_max : float
        Maximum planned Fisher information.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> info_time_from_fisher(fisher_now=250.0, fisher_max=1000.0)
    0.25
    """
    return info_time_from_ratio(fisher_now, fisher_max)


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


class InformationTime(LedgerOp):
    """
    Insert information-time record from sample counts.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    out_id : str
        ID of InformationTimeRecord to create.
    control : BinomialArmSnapshot
        Record containing cumulative counts for the control arm.
    variants : Sequence[BinomialArmSnapshot]
        Record(s) containing cumulative counts for the comparison arm(s).
    planned_max_n : int
        Maximum total sample size.
    """

    out_id: str
    control: BinomialArmSnapshot
    variants: tuple[BinomialArmSnapshot, ...]
    planned_max_n: int

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        control: BinomialArmSnapshot,
        variants: BinomialArmSnapshot | Sequence[BinomialArmSnapshot],
        planned_max_n: int,
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            control=control,
            variants=tuple(variants) if isinstance(variants, Sequence) else (variants,),
            planned_max_n=planned_max_n,
        )

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        if not self.variants:
            raise ValueError("At least one variant record is required.")

        control_payload = self.control.latest_payload()
        variant_payloads = [rec.latest_payload() for rec in self.variants]

        trial_control = int(control_payload["trial"])
        trial_variants = sum(int(payload["trial"]) for payload in variant_payloads)
        n_total = trial_control + trial_variants

        t = info_time_from_sample_size(n_current=n_total, n_max=int(self.planned_max_n))
        out.insert({"info_time": float(t)})


class InformationTimeFromRatio(LedgerOp):
    """Insert t = info_now / info_max."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        info_now: float,
        info_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger, out_id=out_id, info_now=info_now, info_max=info_max, look=look
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_ratio(
            info_now=float(getattr(self, "info_now")),
            info_max=float(getattr(self, "info_max")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromVariance(LedgerOp):
    """Insert t = var_target / var_now."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        var_now: float,
        var_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger, out_id=out_id, var_now=var_now, var_target=var_target, look=look
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_variance(
            var_now=float(getattr(self, "var_now")),
            var_target=float(getattr(self, "var_target")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromSD(LedgerOp):
    """Insert t = (sd_target^2) / (sd_now^2)."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        sd_now: float,
        sd_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            ledger, out_id=out_id, sd_now=sd_now, sd_target=sd_target, look=look
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_sd(
            sd_now=float(getattr(self, "sd_now")),
            sd_target=float(getattr(self, "sd_target")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)
