import json
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Iterable,
    Protocol,
    Self,
    Tuple,
    Type,
    TypeVar,
)

from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from aidial_sdk.exceptions import DeploymentNotFoundError
from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.upstream_config import get_compatible_model_id
from aidial_adapter_bedrock.utils.env import get_str_dict
from aidial_adapter_bedrock.utils.log_config import app_logger as log


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
    The reference Bedrock deployment which is known to share
    the same API as `upstream_deployment_id`.
    """

    @classmethod
    def supported(cls, *, upstream: _T) -> Self:
        return cls(
            upstream_deployment_id=upstream.value,
            reference_deployment_id=upstream,
        )

    def compat(self, deployment_id: str) -> "AdapterDeployment[_T]":
        return AdapterDeployment(
            upstream_deployment_id=deployment_id,
            reference_deployment_id=self.reference_deployment_id,
        )

    def clone(self, reference_deployment_id: _R) -> "AdapterDeployment[_R]":
        return AdapterDeployment(
            upstream_deployment_id=self.upstream_deployment_id,
            reference_deployment_id=reference_deployment_id,
        )


COMPATIBILITY_MAPPING = get_str_dict("COMPATIBILITY_MAPPING")


def resolve_upstream(
    cls: Type[_T], request: FromRequestDeploymentMixin
) -> AdapterDeployment[_T]:
    reference_model_from_upstream = get_compatible_model_id(request)

    upstream_deployment_id = request.original_request.path_params[
        "deployment_id"
    ]

    reference_model_from_compat_mapping = COMPATIBILITY_MAPPING.get(
        upstream_deployment_id
    )

    reference_model = (
        reference_model_from_upstream
        or reference_model_from_compat_mapping
        or upstream_deployment_id
    )

    reference_deployment_id: _T | None = cls.from_string(reference_model)

    if reference_deployment_id is None:
        raise DeploymentNotFoundError(
            f"The deployment id {reference_model!r} is unknown. "
            "It isn't one of the supported deployment ids. "
            "Either fix it if it's a typo, or "
            "set upstreams[*].extraData.compatible_model_id configuration field "
            f"equal to the one of the supported deployment ids compatible with {reference_model!r}."
        )

    return AdapterDeployment[_T](
        upstream_deployment_id=upstream_deployment_id,
        reference_deployment_id=reference_deployment_id,
    )


AdapterChatCompletionDeployment = AdapterDeployment[ChatCompletionDeployment]
AdapterEmbeddingsDeployment = AdapterDeployment[EmbeddingsDeployment]


class AdapterDeployments(BaseModel):
    chat_completions: Dict[str, AdapterChatCompletionDeployment]
    embeddings: Dict[str, AdapterEmbeddingsDeployment]

    @classmethod
    def create(cls, *, compat_mapping: Dict[str, str]) -> "AdapterDeployments":

        chat_completions = set(ChatCompletionDeployment.deployments())
        embeddings = set(EmbeddingsDeployment.deployments())

        for deployment_id, supported_id in compat_mapping.items():
            if deployment_id in chat_completions or deployment_id in embeddings:
                log.warning(
                    f"{deployment_id!r} deployment is already natively supported by the adapter, but it is also mapped to {supported_id!r} in the COMPATIBILITY_MAPPING variable. "
                    f"To avoid this warning and ensure you retain all features of {deployment_id!r}, remove it from the mapping. "
                    f"Otherwise, you may lose features that exist in {deployment_id!r} but are missing in {supported_id!r}."
                )

                if (
                    deployment_id in chat_completions
                    and supported_id in embeddings
                ):
                    raise ValueError(
                        f"The chat completion deployment {deployment_id!r} is mapped onto the embeddings deployment {supported_id!r}"
                    )

                if (
                    deployment_id in embeddings
                    and supported_id in chat_completions
                ):
                    raise ValueError(
                        f"The embeddings deployment {deployment_id!r} is mapped onto the chat completion deployment {supported_id!r}"
                    )

        cross_region_mapping = (
            ChatCompletionDeployment.create_cross_region_inference_mapping()
        )
        compat_mapping = cross_region_mapping | compat_mapping

        compat_mapping, chat_completions = _create_deployments(
            compat_mapping, ChatCompletionDeployment
        )
        compat_mapping, embeddings = _create_deployments(
            compat_mapping, EmbeddingsDeployment
        )

        if compat_mapping:
            raise ValueError(
                f"None of the values in the following compatibility mapping corresponds to a Bedrock deployment supported by the adapter: {json.dumps(compat_mapping)}. "
                f"Remap the deployments to the supported Bedrock deployments to fix the error."
            )

        ret = cls(chat_completions=chat_completions, embeddings=embeddings)

        log.debug(f"Adapter deployments: {ret.json()}")

        return ret


def _create_deployments(
    compat_mapping: Dict[str, str], upstream_deployments: Iterable[_T]
) -> Tuple[Dict[str, str], Dict[str, AdapterDeployment[_T]]]:
    compat_mapping = compat_mapping.copy()

    supported: Dict[str, AdapterDeployment[_T]] = {}
    for upstream in upstream_deployments:
        supported[upstream.value] = AdapterDeployment.supported(
            upstream=upstream
        )

    compat: Dict[str, AdapterDeployment[_T]] = {}
    for deployment_id, supported_deployment_id in list(compat_mapping.items()):
        if (
            supported_deployment := supported.get(supported_deployment_id)
        ) is None:
            continue

        compat_mapping.pop(deployment_id)
        compat[deployment_id] = supported_deployment.compat(deployment_id)

    return compat_mapping, supported | compat
