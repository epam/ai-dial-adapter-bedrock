from pydantic import BaseModel


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"
