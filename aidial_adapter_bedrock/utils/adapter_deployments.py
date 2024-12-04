import json
from typing import Dict, Generic, Iterable, Self, TypeVar

from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import (
    CHAT_COMPLETION_REDIRECTS,
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_D = TypeVar("_D")
_T = TypeVar("_T")


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
            adapter_deployment_id=deployment_id or str(upstream),
            upstream_deployment_id=str(upstream),
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

        chat_completions = _create_deployments(
            compat_mapping,
            ChatCompletionDeployment,
            redirects=CHAT_COMPLETION_REDIRECTS,
        )
        embeddings = _create_deployments(compat_mapping, EmbeddingsDeployment)

        if compat_mapping:
            raise ValueError(
                f"None of the values in the following compatibility dictionary corresponds to a Bedrock deployment supported by the adapter: {json.dumps(compat_mapping)}. "
                f"Remap the deployments to the supported Bedrock deployments to fix the error."
            )

        ret = cls(chat_completions=chat_completions, embeddings=embeddings)

        log.debug(f"Adapter deployments: {ret.json()}")

        return ret


def _create_deployments(
    compat_mapping: Dict[str, str],
    upstream_deployments: Iterable[_D],
    *,
    redirects: Dict[_D, _D] = {},
) -> Dict[str, AdapterDeployment[_D]]:

    supported: Dict[str, AdapterDeployment[_D]] = {}
    for upstream in upstream_deployments:
        deployment_id = str(upstream)
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

        if deployment_id in supported:
            log.warning(
                f"{deployment_id!r} is one of the Bedrock deployments supported by the adapter already. "
                f"Remove {deployment_id!r} from the compatibility mapping to avoid the warning."
            )

        compat_mapping.pop(deployment_id)
        compat[deployment_id] = supported_deployment.compat(deployment_id)

    return supported | compat
