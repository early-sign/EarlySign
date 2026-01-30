from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class AutoNameMixin(BaseModel):
    """
    Mixin to automatically populate the 'name' field with the fully qualified class name
    if it is not provided.
    """

    name: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def default_name_pre(cls, data: Any) -> Any:
        # Handle dict input
        if isinstance(data, dict):
            if "name" not in data or not data["name"]:
                data["name"] = f"{cls.__module__}.{cls.__name__}"
        return data

    @model_validator(mode="after")
    def default_name_post(self) -> Self:
        # Handle object init missing explicit name (if default was used)
        if not self.name:
            self.name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return self
