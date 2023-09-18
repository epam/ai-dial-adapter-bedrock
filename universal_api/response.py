from typing import List, Literal

from pydantic import BaseModel


class ModelObject(BaseModel):
    object: Literal["model"] = "model"
    id: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]
