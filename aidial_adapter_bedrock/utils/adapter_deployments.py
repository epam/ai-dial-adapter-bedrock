import json
from typing import Dict, Generic, Iterable, Self, TypeVar

from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import (
    CHAT_COMPLETION_REDIRECTS,
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.utils.env import get_str_dict
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_D = TypeVar("_D")
_T = TypeVar("_T")


class AdapterDeployment(BaseModel, Generic[_D]):
    adapter_deployment_id: str
    """
    Deployment id under which the model is served by the Adapter
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
    the same API as self.bedrock_model_id.
    """

    @classmethod
    def static(cls, *, deployment_id: str | None = None, upstream: _D) -> Self:
        return cls(
            adapter_deployment_id=deployment_id or str(upstream),
            upstream_deployment_id=str(upstream),
            reference_deployment_id=upstream,
        )

    def dynamic(self, deployment_id: str) -> "AdapterDeployment[_D]":
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


_API_MAPPING_NAME = "API_MAPPING"


class AdapterDeployments(BaseModel):
    chat_completions: Dict[str, AdapterChatCompletionDeployment]
    embeddings: Dict[str, AdapterEmbeddingsDeployment]

    @classmethod
    def create(cls) -> "AdapterDeployments":

        api_mapping = get_str_dict(_API_MAPPING_NAME)
        chat_completions = _create_deployments(
            api_mapping,
            ChatCompletionDeployment,
            redirects=CHAT_COMPLETION_REDIRECTS,
        )
        embeddings = _create_deployments(api_mapping, EmbeddingsDeployment)

        if api_mapping:
            raise ValueError(
                f"None of the values in the following {_API_MAPPING_NAME} dictionary maps to a Bedrock deployment known to the Adapter: {json.dumps(api_mapping)}. "
                f"Remap the deployments to the Bedrock deployments known to the Adapter to fix the error."
            )

        ret = cls(chat_completions=chat_completions, embeddings=embeddings)

        log.debug(f"Adapter deployments: {ret.json()}")

        return ret


def _create_deployments(
    api_mapping: Dict[str, str],
    upstream_deployments: Iterable[_D],
    *,
    redirects: Dict[_D, _D] = {},
) -> Dict[str, AdapterDeployment[_D]]:

    ret: Dict[str, AdapterDeployment[_D]] = {}
    for upstream in upstream_deployments:
        deployment_id = str(upstream)
        ret[deployment_id] = AdapterDeployment.static(
            deployment_id=deployment_id,
            upstream=redirects.get(upstream, upstream),
        )

    for deployment_id, reference_deployment_id in list(api_mapping.items()):
        if (deployment := ret.get(reference_deployment_id)) is None:
            continue

        api_mapping.pop(deployment_id)

        if deployment_id in ret:
            log.warning(
                f"{deployment_id!r} is one of the Bedrock deployments natively supported by the Adapter. "
                f"Remove {deployment_id!r} from the {_API_MAPPING_NAME} env variable to avoid the warning."
            )

        ret[deployment_id] = deployment.dynamic(deployment_id)

    return ret
