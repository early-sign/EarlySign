from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseES3Model(BaseModel):
    """
    Base Pydantic model for all EarlySign Standard Schema (ES3) entities.
    Enforces strict data integrity by forbidding extra fields and validating upon assignment.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class Log(BaseES3Model):
    """
    Base log/event model for metadata inclusion.
    """


class Metadata(BaseES3Model):
    """
    Metadata for EarlySign events and records.
    """

    ES3_version: str | None = Field(
        "v1.0.0", description='Schema version (e.g., "v1.0.0")'
    )


class TaskSpec(BaseES3Model):
    """
    Base class for Problem Definition.
    Concrete protocols should extend this (e.g., GST.TaskSpec).
    """

    kind: str = Field(..., description="The kind of task (discriminator).")


class MethodSpec(BaseES3Model):
    """
    Base class for Operational Method.
    Concrete protocols should extend this (e.g., GST.MethodSpec).
    """

    kind: str = Field(..., description="The kind of method (discriminator).")


class ArmStructureBase(BaseES3Model):
    kind: str


class SingleArm(ArmStructureBase):
    """
    A single-arm trial design.
    """

    kind: Literal["single"] = "single"
    arm_name: str


class TwoArmComparison(ArmStructureBase):
    """
    A classic two-arm comparison (e.g., Treatment vs. Control).
    """

    kind: Literal["two_arm"] = "two_arm"
    control_arm_name: str
    treatment_arm_name: str
    allocation_ratios: dict[str, float] | None = Field(
        None,
        description="Mapping of treatment arm name to its ratio relative to control. Key must match treatment_arm_name.",
    )


class MultiArmComparison(ArmStructureBase):
    """
    A multi-arm comparison against a common control.
    """

    kind: Literal["multi_arm"] = "multi_arm"
    control_arm_name: str
    treatment_arm_names: list[str]
    allocation_ratios: dict[str, float] | None = Field(
        None,
        description="Mapping of treatment arm names to their ratio relative to control (n_treatment / n_control). If None, equal allocation is assumed.",
    )


ArmStructure = Annotated[
    SingleArm | TwoArmComparison | MultiArmComparison,
    Field(
        discriminator="kind",
        description="Structural definition of study arms and their roles.",
    ),
]


class Protocol(BaseES3Model):
    """
    The Root Protocol Container.
    Defines the high-level structure of any EarlySign protocol.
    """

    name: str = Field(..., description="The name of the protocol.")
    task: TaskSpec = Field(..., description="The problem definition.")
    method: MethodSpec = Field(..., description="The operational method.")
