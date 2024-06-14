from pydantic import BaseModel


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"


class ExtraAllowModel(BaseModel):
    class Config:
        extra = "allow"

    def get_extra_fields(self):
        defined_fields = set(self.__fields__.keys())
        all_fields = set(self.__dict__.keys())
        extra_fields = all_fields - defined_fields
        return {field: getattr(self, field) for field in extra_fields}
