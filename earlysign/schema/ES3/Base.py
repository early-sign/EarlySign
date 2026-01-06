from typing import Literal

from pydantic import BaseModel


class TaskSpec(BaseModel):
    kind: str


class MethodSpec(BaseModel):
    kind: str


class Protocol(BaseModel):
    ES3_version: Literal["v1.0.0"] = "v1.0.0"
    name: str
    task: TaskSpec
    method: MethodSpec
