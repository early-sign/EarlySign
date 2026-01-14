"""
Domain-specific write actions using the Ubiquitous Language.

These are semantic wrappers around Writer.Commit for common operations.
They do NOT return identifiers - trace flows through Read, not Write.
"""

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel

from earlysign.v1.framework.trace import TraceId
from earlysign.v1.framework.writer import Writer

if TYPE_CHECKING:
    from earlysign.v1.framework.session import Session


def Decision(
    session: "Session", decision: BaseModel, trace: Optional[List[TraceId]] = None
) -> None:
    """
    Ubiquitous Language: Records an operational conclusion (e.g., Stop Efficacy).
    """
    Writer.Commit(session, decision, trace=trace)


def UpdateProtocol(
    session: "Session", protocol: BaseModel, trace: Optional[List[TraceId]] = None
) -> None:
    """
    Ubiquitous Language: Records a structural update to the trial design (e.g., SSR).
    """
    Writer.Commit(session, protocol, trace=trace)


def Ingest(session: "Session", batch: BaseModel) -> None:
    """
    Ubiquitous Language: Records new raw evidence (Observations).

    Observations are the start of lineage, so they have empty trace.
    """
    Writer.Commit(session, batch, trace=[])  # Root of lineage
