from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel

from earlysign.v1.framework.trace import TraceHash
from earlysign.v1.framework.write_models import WriteModel

if TYPE_CHECKING:
    from earlysign.v1.framework.session import Session


def Decision(
    session: "Session", decision: BaseModel, trace: Optional[List[TraceHash]] = None
) -> TraceHash:
    """
    Ubiquitous Language: Records an operational conclusion (e.g., Stop Efficacy).
    """
    # Logic to record decision with provenance
    return WriteModel.Commit(session, decision, trace=trace)


def UpdateProtocol(
    session: "Session", protocol: BaseModel, trace: Optional[List[TraceHash]] = None
) -> TraceHash:
    """
    Ubiquitous Language: Records a structural update to the trial design (e.g., SSR).
    """
    # Logic to record protocol update
    return WriteModel.Commit(session, protocol, trace=trace)


def Ingest(session: "Session", batch: BaseModel) -> TraceHash:
    """
    Ubiquitous Language: Records new raw evidence (Observations).
    """
    # Logic to append raw evidence
    return WriteModel.Commit(
        session, batch, trace=[]
    )  # Observations are the start of lineage
