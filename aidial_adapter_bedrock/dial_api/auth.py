from typing import Mapping, Optional

from pydantic import BaseModel


class Auth(BaseModel):
    name: str
    value: str

    @property
    def headers(self) -> dict[str, str]:
        return {self.name: self.value}

    @classmethod
    def create_from_headers(
        cls, name: str, headers: Mapping[str, str]
    ) -> Optional["Auth"]:
        value = headers.get(name)
        if value is None:
            return None
        return cls(name=name, value=value)


def get_auth(headers: Mapping[str, str]) -> Auth:
    auth = Auth.create_from_headers(
        "authorization", headers
    ) or Auth.create_from_headers("api-key", headers)

    if auth is None:
        raise Exception(
            "No auth method found. Either authorization or "
            "api-key header must be set in the request"
        )

    return auth
