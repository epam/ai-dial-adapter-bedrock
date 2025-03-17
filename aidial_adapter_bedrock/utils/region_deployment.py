from __future__ import annotations

from enum import Enum
from typing import Dict, Generic, Protocol, Self, TypeVar


# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
class InferenceRegion(Enum):
    US = "us"
    EU = "eu"
    APAC = "apac"


_Origin = TypeVar("_Origin", bound=Enum, covariant=True)


class RegionDeployment(Protocol, Generic[_Origin]):
    @property
    def origin(self) -> _Origin: ...

    @property
    def value(self) -> str: ...


class DeploymentVariant(Generic[_Origin]):
    _origin: _Origin
    _value: str

    def __init__(self, origin: _Origin, value: str) -> None:
        self._origin = origin
        self._value = value

    @property
    def origin(self) -> _Origin:
        return self._origin

    @property
    def value(self) -> str:
        return self._value


class RegionInferenceDeployment(Enum):
    @property
    def origin(self) -> Self:
        return self

    @property
    def US(self) -> RegionDeployment[Self]:
        return self._create_region_variant(InferenceRegion.US)

    @property
    def EU(self) -> RegionDeployment[Self]:
        return self._create_region_variant(InferenceRegion.EU)

    @property
    def APAC(self) -> RegionDeployment[Self]:
        return self._create_region_variant(InferenceRegion.APAC)

    def _create_region_variant(
        self, region: InferenceRegion
    ) -> RegionDeployment[Self]:
        return DeploymentVariant(self, self._get_region_variant(region))

    def _get_region_variant(self, region: InferenceRegion) -> str:
        return f"{region.value}.{self.value}"

    def _is_region_variant(self, region: InferenceRegion) -> bool:
        return self.value.startswith(f"{region.value}.")

    def _cross_region_inference_mapping(self) -> Dict[str, str]:
        if any(self._is_region_variant(region) for region in InferenceRegion):
            return {}

        return {
            self._get_region_variant(region): self.value
            for region in InferenceRegion
        }

    @classmethod
    def create_cross_region_inference_mapping(cls) -> Dict[str, str]:
        return {
            k: v
            for deployment in cls
            for k, v in deployment._cross_region_inference_mapping().items()
        }
