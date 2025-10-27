"""Information time records for group sequential analyses."""

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
