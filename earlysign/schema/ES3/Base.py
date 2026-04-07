from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)

metamodel_version = "1.7.0"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias=True,
        validate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        strict=False,
    )


class LinkMLMeta(RootModel[dict[str, Any]]):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.root, key)

    def __getitem__(self, key: str) -> Any:
        return self.root[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.root[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta(
    {
        "default_prefix": "https://earlysign.io/schema/ES3/base/",
        "default_range": "string",
        "description": "Core base classes for the EarlySign Standard Schema (ES3). "
        "Defines the root protocol container, base task/method specs, "
        "arm structure types, and log events.",
        "id": "https://earlysign.io/schema/ES3/base",
        "imports": ["linkml:types"],
        "name": "es3_base",
        "prefixes": {
            "linkml": {
                "prefix_prefix": "linkml",
                "prefix_reference": "https://w3id.org/linkml/",
            }
        },
        "source_file": "earlysign/schema/ES3/base.yaml",
        "title": "EarlySign Standard Schema (ES3) - Base",
    }
)


class Log(ConfiguredBaseModel):
    """
    Base log/event model for metadata inclusion.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    pass


class Metadata(ConfiguredBaseModel):
    """
    Metadata for EarlySign events and records.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    ES3_version: Optional[str] = Field(
        default="v1.0.0",
        description="""Schema version (e.g., \"v1.0.0\")""",
        json_schema_extra={
            "linkml_meta": {"domain_of": ["Metadata"], "ifabsent": "string(v1.0.0)"}
        },
    )


class TaskSpec(ConfiguredBaseModel):
    """
    Base class for Problem Definition. Concrete protocols should extend this (e.g., GST.TaskSpec).
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    kind: str = Field(
        default=...,
        description="""The kind of task (discriminator).""",
        json_schema_extra={
            "linkml_meta": {
                "domain_of": [
                    "TaskSpec",
                    "MethodSpec",
                    "ArmStructureBase",
                    "SingleArm",
                    "TwoArmComparison",
                    "MultiArmComparison",
                ]
            }
        },
    )


class MethodSpec(ConfiguredBaseModel):
    """
    Base class for Operational Method. Concrete protocols should extend this (e.g., GST.MethodSpec).
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    kind: str = Field(
        default=...,
        description="""The kind of method (discriminator).""",
        json_schema_extra={
            "linkml_meta": {
                "domain_of": [
                    "TaskSpec",
                    "MethodSpec",
                    "ArmStructureBase",
                    "SingleArm",
                    "TwoArmComparison",
                    "MultiArmComparison",
                ]
            }
        },
    )


class Protocol(ConfiguredBaseModel):
    """
    The Root Protocol Container. Defines the high-level structure of any EarlySign protocol.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    name: str = Field(
        default=...,
        description="""The name of the protocol.""",
        json_schema_extra={"linkml_meta": {"domain_of": ["Protocol"]}},
    )
    task: TaskSpec = Field(
        default=...,
        description="""The problem definition.""",
        json_schema_extra={"linkml_meta": {"domain_of": ["Protocol"]}},
    )
    method: MethodSpec = Field(
        default=...,
        description="""The operational method.""",
        json_schema_extra={"linkml_meta": {"domain_of": ["Protocol"]}},
    )


class ArmStructureBase(ConfiguredBaseModel):
    """
    Internal base for arm structure discriminated union members.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    kind: str = Field(
        default=...,
        json_schema_extra={
            "linkml_meta": {
                "domain_of": [
                    "TaskSpec",
                    "MethodSpec",
                    "ArmStructureBase",
                    "SingleArm",
                    "TwoArmComparison",
                    "MultiArmComparison",
                ]
            }
        },
    )


class SingleArm(ArmStructureBase):
    """
    A single-arm trial design.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    kind: Literal["single"] = Field(
        default="single",
        json_schema_extra={
            "linkml_meta": {
                "domain_of": [
                    "TaskSpec",
                    "MethodSpec",
                    "ArmStructureBase",
                    "SingleArm",
                    "TwoArmComparison",
                    "MultiArmComparison",
                ],
                "equals_string": "single",
                "ifabsent": "string(single)",
            }
        },
    )
    arm_name: str = Field(
        default=...,
        json_schema_extra={"linkml_meta": {"domain_of": ["SingleArm", "ArmStatus"]}},
    )


class TwoArmComparison(ArmStructureBase):
    """
    A classic two-arm comparison (e.g., Treatment vs. Control).
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    kind: Literal["two_arm"] = Field(
        default="two_arm",
        json_schema_extra={
            "linkml_meta": {
                "domain_of": [
                    "TaskSpec",
                    "MethodSpec",
                    "ArmStructureBase",
                    "SingleArm",
                    "TwoArmComparison",
                    "MultiArmComparison",
                ],
                "equals_string": "two_arm",
                "ifabsent": "string(two_arm)",
            }
        },
    )
    control_arm_name: str = Field(
        default=...,
        json_schema_extra={
            "linkml_meta": {"domain_of": ["TwoArmComparison", "MultiArmComparison"]}
        },
    )
    treatment_arm_name: str = Field(
        default=...,
        json_schema_extra={"linkml_meta": {"domain_of": ["TwoArmComparison"]}},
    )
    allocation_ratios: Optional[list[float]] = Field(
        default=None,
        description="""Mapping of treatment arm name to its ratio relative to control. Key must match treatment_arm_name.""",
        json_schema_extra={
            "linkml_meta": {
                "annotations": {
                    "python_range": {
                        "tag": "python_range",
                        "value": "Optional[Dict[str, float]]",
                    }
                },
                "domain_of": ["TwoArmComparison", "MultiArmComparison"],
            }
        },
    )


class MultiArmComparison(ArmStructureBase):
    """
    A multi-arm comparison against a common control.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    kind: Literal["multi_arm"] = Field(
        default="multi_arm",
        json_schema_extra={
            "linkml_meta": {
                "domain_of": [
                    "TaskSpec",
                    "MethodSpec",
                    "ArmStructureBase",
                    "SingleArm",
                    "TwoArmComparison",
                    "MultiArmComparison",
                ],
                "equals_string": "multi_arm",
                "ifabsent": "string(multi_arm)",
            }
        },
    )
    control_arm_name: str = Field(
        default=...,
        json_schema_extra={
            "linkml_meta": {"domain_of": ["TwoArmComparison", "MultiArmComparison"]}
        },
    )
    treatment_arm_names: list[str] = Field(
        default=...,
        json_schema_extra={"linkml_meta": {"domain_of": ["MultiArmComparison"]}},
    )
    allocation_ratios: Optional[list[float]] = Field(
        default=None,
        description="""Mapping of treatment arm names to their ratio relative to control (n_treatment / n_control). If None, equal allocation is assumed.""",
        json_schema_extra={
            "linkml_meta": {
                "annotations": {
                    "python_range": {
                        "tag": "python_range",
                        "value": "Optional[Dict[str, float]]",
                    }
                },
                "domain_of": ["TwoArmComparison", "MultiArmComparison"],
            }
        },
    )


# Structural definition of study arms and their roles.
ArmStructure = Union["SingleArm", "TwoArmComparison", "MultiArmComparison"]


class ArmMetrics(Log):
    """
    Metrics for a single arm in a sequential test.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    total: int = Field(
        default=..., json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    ci_lower: Optional[float] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    ci_upper: Optional[float] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    quantile_estimate: Optional[float] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    mean: Optional[float] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    variance: Optional[float] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    successes: Optional[int] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )
    p_hat: Optional[float] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["ArmMetrics"]}}
    )


class ArmStatus(Log):
    """
    Status wrapper for arm metrics.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    arm_name: str = Field(
        default=...,
        json_schema_extra={"linkml_meta": {"domain_of": ["SingleArm", "ArmStatus"]}},
    )
    metrics: ArmMetrics = Field(
        default=..., json_schema_extra={"linkml_meta": {"domain_of": ["ArmStatus"]}}
    )
    is_active: bool = Field(
        default=..., json_schema_extra={"linkml_meta": {"domain_of": ["ArmStatus"]}}
    )


class Scoreboard(Log):
    """
    Projection of all arm metrics.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    arms: Optional[dict[str, ArmStatus]] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["Scoreboard"]}}
    )


class LookResult(Log):
    """
    Base class for intermediate look results.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://earlysign.io/schema/ES3/base"}
    )

    look: Optional[int] = Field(
        default=None, json_schema_extra={"linkml_meta": {"domain_of": ["LookResult"]}}
    )


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Log.model_rebuild()
Metadata.model_rebuild()
TaskSpec.model_rebuild()
MethodSpec.model_rebuild()
Protocol.model_rebuild()
ArmStructureBase.model_rebuild()
SingleArm.model_rebuild()
TwoArmComparison.model_rebuild()
MultiArmComparison.model_rebuild()
ArmMetrics.model_rebuild()
ArmStatus.model_rebuild()
Scoreboard.model_rebuild()
LookResult.model_rebuild()
