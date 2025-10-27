"""Statistic record definitions for group sequential execution."""

from earlysign.framework.records import LedgerRecord, QueryMixin


class WaldZStatisticRecord(LedgerRecord, QueryMixin):
    """Wald Z statistic record (scheme-agnostic)."""

    schema = {
        "wald_z": (float, ...),
    }
