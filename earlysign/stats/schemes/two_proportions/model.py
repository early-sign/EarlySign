"""
earlysign.schemes.two_proportions.model
=======================================

Typed payloads for the *two-sample binomial* scheme.

- `TwoPropObsBatch`: single observation batch
- `TwoPropDesign`: design metadata (optional)
- TypedDict payload contracts for statistics/criteria (mypy-friendly)

Examples
--------
>>> from earlysign.core.ledger import PayloadRegistry
>>> from earlysign.stats.schemes.two_proportions.model import TwoPropObsBatch
>>> PayloadRegistry.register("TwoPropObsBatch", lambda d: TwoPropObsBatch(**d))
>>> from earlysign.core.ledger import PayloadRegistry as PR
>>> isinstance(PR.decode("TwoPropObsBatch", {"nA":1,"nB":2,"mA":0,"mB":1}), TwoPropObsBatch)
True
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, TypedDict


# --- Typed payloads used in ledger records (mypy-friendly) ---


class WaldZPayload(TypedDict):
    z: float
    se: float
    nA: int
    nB: int
    mA: int
    mB: int
    pA_hat: float
    pB_hat: float


class GstBoundaryPayload(TypedDict):
    upper: float
    lower: float
    info_time: float
    alpha_i: float


# --- Typed objects (optional decoder targets) ---


@dataclass(frozen=True)
class TwoPropObsBatch:
    nA: int
    nB: int
    mA: int
    mB: int


@dataclass(frozen=True)
class TwoPropDesign:
    design_id: str
    version: int
    alpha: float
    power: float
    pA: float
    pB: float
    looks: int
    spending: str
    t_grid: Tuple[float, ...]
    note: str = ""
