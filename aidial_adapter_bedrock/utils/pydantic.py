from pydantic import BaseModel


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"


class ExtraAllowModel(BaseModel):
    class Config:
        extra = "allow"


class AnyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
