import json
from enum import Enum
from typing import Dict, Generic, Iterable, Self, Tuple, TypeVar

from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import (
    CHAT_COMPLETION_REDIRECTS,
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_D = TypeVar("_D", bound=Enum)
_T = TypeVar("_T", bound=Enum)


class AdapterDeployment(BaseModel, Generic[_D]):
    adapter_deployment_id: str
    """
    The deployment id under which the model is served by the Adapter
    at the route /openai/deployments/{deployment_id}/(chat/completions|embeddings)
    """

    upstream_deployment_id: str
    """
    The deployment id of the corresponding Bedrock model.
    The upstream request to the Bedrock service will use this deployment id.
    """

    reference_deployment_id: _D
    """
    The reference Bedrock deployment which is known to share
    the same API as `upstream_deployment_id`.
    """

    @classmethod
    def supported(
        cls, *, deployment_id: str | None = None, upstream: _D
    ) -> Self:
        return cls(
            adapter_deployment_id=deployment_id or upstream.value,
            upstream_deployment_id=upstream.value,
            reference_deployment_id=upstream,
        )

    def compat(self, deployment_id: str) -> "AdapterDeployment[_D]":
        return AdapterDeployment(
            adapter_deployment_id=deployment_id,
            upstream_deployment_id=deployment_id,
            reference_deployment_id=self.reference_deployment_id,
        )

    def clone(self, reference_deployment_id: _T) -> "AdapterDeployment[_T]":
        return AdapterDeployment(
            adapter_deployment_id=self.adapter_deployment_id,
            upstream_deployment_id=self.upstream_deployment_id,
            reference_deployment_id=reference_deployment_id,
        )


AdapterChatCompletionDeployment = AdapterDeployment[ChatCompletionDeployment]
AdapterEmbeddingsDeployment = AdapterDeployment[EmbeddingsDeployment]


class AdapterDeployments(BaseModel):
    chat_completions: Dict[str, AdapterChatCompletionDeployment]
    embeddings: Dict[str, AdapterEmbeddingsDeployment]

    @classmethod
    def create(cls, *, compat_mapping: Dict[str, str]) -> "AdapterDeployments":
        cross_region_mapping = (
            ChatCompletionDeployment.create_cross_region_inference_mapping()
        )
        compat_mapping = cross_region_mapping | compat_mapping

        chat_completions = {e.value for e in ChatCompletionDeployment}
        embeddings = {e.value for e in EmbeddingsDeployment}

        for deployment_id, supported_id in compat_mapping.items():
            if deployment_id in chat_completions or deployment_id in embeddings:
                log.warning(
                    f"{deployment_id!r} is one of the Bedrock deployments supported by the adapter already. "
                    f"Remove {deployment_id!r} from the compatibility mapping to avoid the warning, otherwise you are losing the features present in the former deployment and missing from the latter."
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

        compat_mapping, chat_completions = _create_deployments(
            compat_mapping,
            ChatCompletionDeployment,
            redirects=CHAT_COMPLETION_REDIRECTS,
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
    compat_mapping: Dict[str, str],
    upstream_deployments: Iterable[_D],
    *,
    redirects: Dict[_D, _D] | None = None,
) -> Tuple[Dict[str, str], Dict[str, AdapterDeployment[_D]]]:
    compat_mapping = compat_mapping.copy()
    redirects = redirects or {}

    supported: Dict[str, AdapterDeployment[_D]] = {}
    for upstream in upstream_deployments:
        deployment_id = upstream.value
        supported[deployment_id] = AdapterDeployment.supported(
            deployment_id=deployment_id,
            upstream=redirects.get(upstream, upstream),
        )

    compat: Dict[str, AdapterDeployment[_D]] = {}
    for deployment_id, supported_deployment_id in list(compat_mapping.items()):
        if (
            supported_deployment := supported.get(supported_deployment_id)
        ) is None:
            continue

        compat_mapping.pop(deployment_id)
        compat[deployment_id] = supported_deployment.compat(deployment_id)

    return compat_mapping, supported | compat
