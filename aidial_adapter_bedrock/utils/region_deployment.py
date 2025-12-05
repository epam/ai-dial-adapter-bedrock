import re
from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Protocol, Self, TypeVar

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
class InferenceRegion(Enum):
    US = "us"
    EU = "eu"
    APAC = "apac"
    JP = "jp"
    GLOBAL = "global"

    @classmethod
    def prefixes(cls) -> str:
        return ", ".join(p.value for p in cls)


_Origin = TypeVar("_Origin", bound=Enum, covariant=True)


class RegionDeployment(Protocol, Generic[_Origin]):
    @property
    def origin(self) -> _Origin: ...

    @property
    def value(self) -> str: ...


@dataclass(frozen=True)
class DeploymentVariant(Generic[_Origin]):
    origin: _Origin
    value: str


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

    @property
    def variants(self) -> List[str]:
        return [self.value] + [
            self._get_region_variant(region) for region in InferenceRegion
        ]

    @classmethod
    def deployments(cls) -> List[str]:
        """
        Return a list of all regional and non-regional deployments:
        [deployment1, us.deployment1, eu.deployment1, apac.deployment1, deployment2, ...]
        """
        return [v for d in cls for v in d.variants]

    @classmethod
    def from_string(cls, model_id: str) -> Self | None:
        for deployment in cls:
            deployment_id: str = deployment.value
            if model_id.endswith(deployment_id):
                prefix = model_id.removesuffix(deployment_id)
                if prefix == "":
                    return deployment

                if (parsed := _parse_region_prefix(prefix)) is None:
                    continue

                if not isinstance(parsed, InferenceRegion):
                    log.warning(
                        f"{model_id!r} has unexpected cross-region prefix {parsed!r} that "
                        f"doesn't match any of the supported prefixes: {InferenceRegion.prefixes()}."
                    )

                return deployment
        return None


def _parse_region_prefix(s: str) -> InferenceRegion | str | None:
    for region in InferenceRegion:
        if f"{region.value}." == s:
            return region

    if m := re.fullmatch(r"(\w+)\.", s):
        return m.group(1)

    return None
