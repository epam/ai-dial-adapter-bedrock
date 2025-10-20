from enum import Enum
from typing import Dict, Generic, Iterable, List, Protocol, Self, TypeVar


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

    def _get_region_variants(self) -> List[str]:
        if self._is_region_variant():
            return []
        return [self._get_region_variant(region) for region in InferenceRegion]

    def _is_region_variant(self) -> bool:
        return any(
            self.value.startswith(f"{region.value}.")
            for region in InferenceRegion
        )

    def _cross_region_inference_mapping(self) -> Dict[str, str]:
        """
        Return the mapping from regional variants to the original deployment:
            {   us.deployment: deployment,
                eu.deployment: deployment,
                apac.deployment: deployment
            }
        """
        return {variant: self.value for variant in self._get_region_variants()}

    @classmethod
    def create_cross_region_inference_mapping(cls) -> Dict[str, str]:
        """
        Return the mapping from all regional variants to their respective original deployments.
            {   us.deployment1: deployment1,
                eu.deployment1: deployment1,
                apac.deployment1: deployment1,
                us.deployment2: deployment2,
                eu.deployment2: deployment2,
                apac.deployment2: deployment2,
                ...
            }
        """

        return {
            k: v
            for deployment in cls
            for k, v in deployment._cross_region_inference_mapping().items()
        }

    @classmethod
    def deployments(cls) -> Iterable[str]:
        """
        Return a list of all regional and non-regional deployments:
        [deployment1, us.deployment1, eu.deployment1, apac.deployment1, deployment2, ...]
        """
        ret: List[str] = []
        for deployment in cls:
            ret.append(deployment.value)
            ret.extend(deployment._get_region_variants())
        return ret
