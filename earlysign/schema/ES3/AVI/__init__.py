from typing import Literal

from earlysign.schema.ES3.Base import (
    MethodSpec as BaseMethodSpec,
    Protocol as BaseProtocol,
    TaskSpec as BaseTaskSpec,
)


class AVITaskSpec(BaseTaskSpec):
    kind: Literal["anytime_valid"] = "anytime_valid"
    alpha: float
    null_p: float
    alt_p: float


class AVIMethodSpec(BaseMethodSpec):
    kind: Literal["anytime_valid"] = "anytime_valid"
    strategy: str = "mixture"


class Protocol(BaseProtocol):
    task: AVITaskSpec
    method: AVIMethodSpec
