import os
from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel


class Auth(BaseModel):
    @property
    @abstractmethod
    def headers(self) -> dict[str, str]:
        pass


class ApiKey(Auth):
    key: str

    @property
    def headers(self) -> dict[str, str]:
        return {"api-key": self.key}


class AuthToken(Auth):
    token: str

    @property
    def headers(self) -> dict[str, str]:
        return {"authorization": self.token}


def get_auth(jwt_token: Optional[str]) -> Auth:
    if jwt_token is not None:
        return AuthToken(token=jwt_token)

    api_key_var = "DIAL_API_KEY"
    api_key = os.environ.get(api_key_var)
    if api_key is not None:
        return ApiKey(key=api_key)

    raise Exception(
        "No auth method found. Either JWT token must be set in the request "
        f"or {api_key_var} env variable is provided"
    )
