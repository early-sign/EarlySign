"""
Information-time operators (scheme-agnostic).

This module provides multiple estimators of "information time" t in [0, 1].
Each operator writes an InformationTimeRecord that downstream GS components use.

- InformationTime            : t = clip(n_total / N_max) or planned_fractions[look]
- InformationTimeFromRatio   : t = clip(info_now / info_max)              # Fisher info, precision, etc.
- InformationTimeFromVariance: t = clip(var_target / var_now)             # information ∝ 1/variance
- InformationTimeFromSD      : t = clip((sd_target**2) / (sd_now**2))     # SD-based variant
- InformationTimeFromFisher  : t = clip(fisher_now / fisher_max)          # explicit Fisher information
"""

from typing import Dict, List, Mapping, Optional, Union

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.records import InformationTimeRecord

# ---------------- helpers (each function has its own doctest) ----------------


def _clip01(x: float) -> float:
    """
    Clip a float into [0, 1].

    Examples
    --------
    >>> _clip01(-0.1)
    0.0
    >>> _clip01(0.25)
    0.25
    >>> _clip01(2.0)
    1.0
    """
    return max(0.0, min(1.0, float(x)))


def compute_information_time(
    n_total: Optional[int] = None,
    N_max: Optional[int] = None,
    *,
    current_look: Optional[int] = None,
    planned_fractions: Optional[Union[Mapping[int, float], List[float]]] = None,
) -> float:
    """
    Compute information time in [0, 1] from counts or planned fractions.

    Examples
    --------
    >>> compute_information_time(n_total=100, N_max=400)
    0.25
    >>> compute_information_time(current_look=2, planned_fractions={1:0.25, 2:0.5, 3:0.75})
    0.5
    """
    if planned_fractions is not None:
        if current_look is None:
            raise ValueError(
                "`current_look` must be provided when `planned_fractions` is used."
            )
        if isinstance(planned_fractions, dict):
            if current_look not in planned_fractions:
                raise KeyError(
                    "look={} not found in `planned_fractions`.".format(current_look)
                )
            return _clip01(planned_fractions[current_look])
        idx = current_look - 1
        if idx < 0 or idx >= len(planned_fractions):
            raise IndexError(
                "`current_look` out of range for `planned_fractions` list."
            )
        return _clip01(planned_fractions[idx])

    if n_total is None or N_max is None:
        raise ValueError(
            "Provide either (n_total, N_max) or (current_look, planned_fractions)."
        )
    if N_max <= 0:
        raise ValueError("`N_max` must be positive.")
    return _clip01(float(n_total) / float(N_max))


def compute_information_time_from_ratio(info_now: float, info_max: float) -> float:
    """
    Generic ratio: t = clip(info_now / info_max).

    Examples
    --------
    >>> compute_information_time_from_ratio(info_now=3.0, info_max=12.0)
    0.25
    """
    if info_max <= 0:
        raise ValueError("`info_max` must be positive.")
    return _clip01(float(info_now) / float(info_max))


def compute_information_time_from_variance(var_now: float, var_target: float) -> float:
    """
    Variance-based timing: information ∝ 1/variance ⇒ t = clip(var_target / var_now).

    Examples
    --------
    >>> compute_information_time_from_variance(var_now=4.0, var_target=1.0)
    0.25
    """
    if var_now <= 0 or var_target <= 0:
        raise ValueError("variances must be positive.")
    return _clip01(float(var_target) / float(var_now))


def compute_information_time_from_sd(sd_now: float, sd_target: float) -> float:
    """
    SD-based timing: information ∝ 1/(sd^2) ⇒ t = clip((sd_target^2)/(sd_now^2)).

    Examples
    --------
    >>> compute_information_time_from_sd(sd_now=2.0, sd_target=1.0)
    0.25
    """
    if sd_now <= 0 or sd_target <= 0:
        raise ValueError("standard deviations must be positive.")
    return _clip01((float(sd_target) ** 2) / (float(sd_now) ** 2))


def compute_information_time_from_fisher(fisher_now: float, fisher_max: float) -> float:
    """
    Fisher-information timing: t = clip(fisher_now / fisher_max).

    Examples
    --------
    >>> compute_information_time_from_fisher(fisher_now=30.0, fisher_max=120.0)
    0.25
    """
    if fisher_max <= 0:
        raise ValueError("`fisher_max` must be positive.")
    return _clip01(float(fisher_now) / float(fisher_max))


# ---------------- operators (derived_records() instantiates outputs) ----------------


class InformationTime(LedgerOperator):
    """
    Insert an information-time row from counts/plan.

    __init__ parameters (stored by the base class)
    ----------------------------------------------
    out_id : str
        ID of the InformationTimeRecord to create (in derived_records()).
    n_total, N_max : Optional[int]
    current_look : Optional[int]
    planned_fractions : Optional[dict[int,float] | list[float]]
    """

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        n_total: Optional[int] = None,
        N_max: Optional[int] = None,
        current_look: Optional[int] = None,
        planned_fractions: Optional[Union[Mapping[int, float], List[float]]] = None,
    ):
        super().__init__(
            scoped,
            out_id=out_id,
            n_total=n_total,
            N_max=N_max,
            current_look=current_look,
            planned_fractions=planned_fractions,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["info"]
        t = compute_information_time(
            n_total=getattr(self, "n_total", None),
            N_max=getattr(self, "N_max", None),
            current_look=getattr(self, "current_look", None),
            planned_fractions=getattr(self, "planned_fractions", None),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "current_look", None) is not None:
            payload["look"] = int(getattr(self, "current_look"))
        out.insert(payload)


class InformationTimeFromRatio(LedgerOperator):
    """Insert t = info_now / info_max (optionally store look)."""

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        info_now: float,
        info_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped, out_id=out_id, info_now=info_now, info_max=info_max, look=look
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["info"]
        t = compute_information_time_from_ratio(
            float(getattr(self, "info_now")), float(getattr(self, "info_max"))
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromVariance(LedgerOperator):
    """Insert t = var_target / var_now (clipped)."""

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        var_now: float,
        var_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped, out_id=out_id, var_now=var_now, var_target=var_target, look=look
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["info"]
        t = compute_information_time_from_variance(
            float(getattr(self, "var_now")), float(getattr(self, "var_target"))
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromSD(LedgerOperator):
    """Insert t = (sd_target^2)/(sd_now^2)."""

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        sd_now: float,
        sd_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped, out_id=out_id, sd_now=sd_now, sd_target=sd_target, look=look
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["info"]
        t = compute_information_time_from_sd(
            float(getattr(self, "sd_now")), float(getattr(self, "sd_target"))
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromFisher(LedgerOperator):
    """Insert t = fisher_now / fisher_max."""

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        fisher_now: float,
        fisher_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped,
            out_id=out_id,
            fisher_now=fisher_now,
            fisher_max=fisher_max,
            look=look,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["info"]
        t = compute_information_time_from_fisher(
            float(getattr(self, "fisher_now")), float(getattr(self, "fisher_max"))
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)
