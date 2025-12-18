from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, Self, TypeVar

from pydantic import BaseModel


class ReadableStrEnum(Protocol):
    @classmethod
    def from_string(cls, model_id: str) -> Self | None: ...

    @property
    def value(self) -> str: ...


if TYPE_CHECKING:
    ReadableStrEnumT = ReadableStrEnum
else:
    from enum import Enum as ReadableStrEnumT


_T = TypeVar("_T", bound=ReadableStrEnumT)
_R = TypeVar("_R", bound=ReadableStrEnumT)


class AdapterDeployment(BaseModel, Generic[_T]):
    upstream_deployment_id: str
    """
    The deployment id of the corresponding Bedrock model.
    The upstream request to the Bedrock service will use this deployment id.
    """

    reference_deployment_id: _T
    """
    The reference Bedrock deployment that is known to share
    the same API as `upstream_deployment_id`.
    """

    def clone(self, deployment: _R) -> AdapterDeployment[_R]:
        return AdapterDeployment(
            upstream_deployment_id=self.upstream_deployment_id,
            reference_deployment_id=deployment,
        )
