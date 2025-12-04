import json
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
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

_UPSTREAM_CONFIG_PATH = (
    "upstreams[*].extraData.compatible_model_id field in the DIAL Core config"
)
_COMPAT_MAPPING_NAME = "COMPATIBILITY_MAPPING env variable"


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

    compatible_deployment_id: _T
    """
    The reference Bedrock deployment that is known to share
    the same API as `upstream_deployment_id`.
    """

    @classmethod
    def supported(cls, deployment: _T) -> Self:
        return cls(
            upstream_deployment_id=deployment.value,
            compatible_deployment_id=deployment,
        )

    def compat(self, upstream_deployment_id: str) -> "AdapterDeployment[_T]":
        return AdapterDeployment(
            upstream_deployment_id=upstream_deployment_id,
            compatible_deployment_id=self.compatible_deployment_id,
        )

    def clone(self, compatible_deployment_id: _R) -> "AdapterDeployment[_R]":
        return AdapterDeployment(
            upstream_deployment_id=self.upstream_deployment_id,
            compatible_deployment_id=compatible_deployment_id,
        )


COMPATIBILITY_MAPPING = get_str_dict("COMPATIBILITY_MAPPING")


def resolve_deployment_from_request(
    cls: Type[_T], request: FromRequestDeploymentMixin
) -> AdapterDeployment[_T]:
    deployment_id = request.original_request.path_params["deployment_id"]
    return resolve_deployment(
        cls,
        upstream_deployment_id=deployment_id,
        compat_mapping=COMPATIBILITY_MAPPING,
        compatible_id_from_upstream=get_compatible_model_id(request),
    )


def resolve_deployment(
    cls: Type[_T],
    *,
    upstream_deployment_id: str,
    compat_mapping: dict[str, str] | None = None,
    compatible_id_from_upstream: str | None = None,
) -> AdapterDeployment[_T]:
    if (
        compatible_id_from_upstream is not None
        and cls.from_string(upstream_deployment_id) is not None
    ):
        log.warning(
            f"{upstream_deployment_id!r} deployment is already natively supported by the adapter, "
            f"but it is also mapped to {compatible_id_from_upstream!r} in {_UPSTREAM_CONFIG_PATH}. "
            f"To avoid this warning and ensure you retain all features of {upstream_deployment_id!r}, "
            "remove the corresponding field. "
            f"Otherwise, you may lose features that exist in {upstream_deployment_id!r} but "
            f"are missing in {compatible_id_from_upstream!r}."
        )

    compatible_id_from_compat_mapping = (compat_mapping or {}).get(
        upstream_deployment_id
    )

    compatible_id = (
        compatible_id_from_upstream
        or compatible_id_from_compat_mapping
        or upstream_deployment_id
    )

    compatible_deployment_id: _T | None = cls.from_string(compatible_id)

    if compatible_deployment_id is None:
        if (
            compatible_id_from_upstream is None
            and compatible_id_from_compat_mapping is None
        ):
            msg = (
                f"The deployment id {compatible_id!r} isn't one of the deployment ids supported by the adapter. "
                f"Either replace it with a supported deployment id, or set {_UPSTREAM_CONFIG_PATH} "
                f"equal to a supported deployment id that is compatible with {compatible_id!r}."
            )
        elif compatible_id_from_upstream is not None:
            msg = (
                f"{compatible_id!r} is declared as a deployment id that is compatible with {upstream_deployment_id!r} via {_UPSTREAM_CONFIG_PATH}. "
                f"However, {compatible_id!r} isn't one of the deployment ids supported by the adapter. "
                f"Replace it with a supported deployment id to avoid this error."
            )
        else:
            msg = (
                f"{compatible_id!r} is declared as a deployment id that is compatible with {upstream_deployment_id!r} via {_COMPAT_MAPPING_NAME}. "
                f"However, {compatible_id!r} isn't one of the deployment ids supported by the adapter. "
                f"Replace it with a supported deployment id to avoid this error."
            )

        raise DeploymentNotFoundError(msg)

    return AdapterDeployment[_T](
        upstream_deployment_id=upstream_deployment_id,
        compatible_deployment_id=compatible_deployment_id,
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
                    f"{deployment_id!r} deployment is already natively supported by the adapter, "
                    f"but it is also mapped to {supported_id!r} in the {_COMPAT_MAPPING_NAME}. "
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

        residual_compat_mapping, chat_completions = _create_deployments(
            ChatCompletionDeployment, compat_mapping
        )
        residual_compat_mapping, embeddings = _create_deployments(
            EmbeddingsDeployment, residual_compat_mapping
        )

        if residual_compat_mapping:
            raise ValueError(
                f"None of the values in the following compatibility mapping corresponds to a "
                f"Bedrock deployment supported by the adapter: {json.dumps(residual_compat_mapping)}. "
                f"Remap the deployments to the supported Bedrock deployments to fix the error."
            )

        ret = cls(chat_completions=chat_completions, embeddings=embeddings)

        log.debug(f"Static compatibility deployments: {ret.json()}")
        if msg := ret._deprecation_warning():
            log.warning(msg)

        return ret._enrich_with_supported_deployments()

    def _deprecation_warning(self) -> str | None:
        deployments = self.embeddings | self.chat_completions
        if not deployments:
            return None

        def create_model_entry(
            deployment: (
                AdapterEmbeddingsDeployment | AdapterChatCompletionDeployment
            ),
        ) -> dict:
            is_chat = isinstance(
                deployment.compatible_deployment_id, ChatCompletionDeployment
            )
            ty = "chat" if is_chat else "embedding"
            endpoint = "chat/completions" if is_chat else "embeddings"
            compatible_id = deployment.compatible_deployment_id.value
            extra = {"compatible_deployment_id": compatible_id}
            return {
                "type": ty,
                "endpoint": f"$ADAPTER_ORIGIN/openai/deployments/{deployment.upstream_deployment_id}/{endpoint}",
                "upstreams": [{"extraData": extra}],
            }

        models, idx = {}, 1
        for deployment in sorted(deployments.values(), key=str):
            models[f"$DIAL_DEPLOYMENT_ID{idx}"] = create_model_entry(deployment)
            idx += 1
        config = {"models": models}

        return (
            f"{_COMPAT_MAPPING_NAME} is deprecated in favour of per-upstream configuration in DIAL Core config. "
            "You may remove the entries from the env variable one-by-one and amend configurations "
            f"for corresponding deployments in the DIAL Core config: {json.dumps(config)}"
        )

    def _enrich_with_supported_deployments(self) -> "AdapterDeployments":
        chat_completions = self.chat_completions.copy()
        for deployment in ChatCompletionDeployment:
            for variant in deployment.variants:
                if variant not in self.chat_completions:
                    chat_completions[variant] = AdapterDeployment(
                        upstream_deployment_id=variant,
                        compatible_deployment_id=deployment,
                    )

        embeddings = self.embeddings.copy()
        for deployment in EmbeddingsDeployment:
            variant = deployment.value
            if variant not in self.chat_completions:
                embeddings[variant] = AdapterDeployment(
                    upstream_deployment_id=variant,
                    compatible_deployment_id=deployment,
                )

        return AdapterDeployments(
            chat_completions=chat_completions, embeddings=embeddings
        )


def _create_deployments(
    cls: Type[_T], compat_mapping: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, AdapterDeployment[_T]]]:
    leftovers: Dict[str, str] = {}
    compat: Dict[str, AdapterDeployment[_T]] = {}

    for deployment_id, supported_deployment_id in compat_mapping.items():
        compatible_deployment_id: _T | None = cls.from_string(
            supported_deployment_id
        )

        if compatible_deployment_id is None:
            leftovers[deployment_id] = supported_deployment_id
        else:
            compat[deployment_id] = AdapterDeployment(
                upstream_deployment_id=deployment_id,
                compatible_deployment_id=compatible_deployment_id,
            )

    return leftovers, compat


def get_static_deployments() -> AdapterDeployments:
    return AdapterDeployments.create(compat_mapping=COMPATIBILITY_MAPPING)
