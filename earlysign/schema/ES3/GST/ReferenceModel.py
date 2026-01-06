from typing import Literal

from pydantic import BaseModel


class ReferenceModelSpec(BaseModel):
    kind: str


class BinaryModel(ReferenceModelSpec):
    kind: Literal["binary"] = "binary"
    test_statistic: Literal["Z", "chi_square", "exact"]
    link_function: Literal["identity", "log", "logit"] = "identity"
    use_canonical_joint_distribution: bool


class ContinuousModel(ReferenceModelSpec):
    kind: Literal["continuous"] = "continuous"
    test_statistic: Literal["Z", "t"]
    variance_assumption: Literal["known", "unknown", "heteroscedastic"]
    use_canonical_joint_distribution: bool


class SurvivalModel(ReferenceModelSpec):
    kind: Literal["time_to_event"] = "time_to_event"
    test_statistic: Literal["log_rank", "cox_ph"]
    use_canonical_joint_distribution: bool
