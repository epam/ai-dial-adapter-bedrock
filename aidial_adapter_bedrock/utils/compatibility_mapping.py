import json
import logging
from typing import Any, Dict

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.utils.adapter_deployment import AdapterDeployment
from aidial_adapter_bedrock.utils.env import get_str_dict
from aidial_adapter_bedrock.utils.log_config import app_logger as log

COMPATIBILITY_MAPPING = get_str_dict("COMPATIBILITY_MAPPING")
COMPAT_MAPPING_NAME = "COMPATIBILITY_MAPPING env variable"


def _validate_compat_mapping(compat_mapping: Dict[str, str]):
    log.debug(f"Compatibility mapping: {json.dumps(compat_mapping)}")

    chat_completions = set(ChatCompletionDeployment.deployments())
    embeddings = set(EmbeddingsDeployment.deployments())

    for unsupported_id, supported_id in compat_mapping.items():
        if unsupported_id in chat_completions or unsupported_id in embeddings:
            log.warning(
                f"{unsupported_id!r} deployment is already natively supported by the adapter, "
                f"but it is also mapped to {supported_id!r} in the {COMPAT_MAPPING_NAME}. "
                f"To avoid this warning and ensure you retain all features of {unsupported_id!r}, remove it from the mapping. "
                f"Otherwise, you may lose features that exist in {unsupported_id!r} but are missing in {supported_id!r}."
            )

            if (
                unsupported_id in chat_completions
                and supported_id in embeddings
            ):
                raise ValueError(
                    f"The chat completion deployment {unsupported_id!r} is mapped onto the embeddings deployment {supported_id!r}"
                )

            if (
                unsupported_id in embeddings
                and supported_id in chat_completions
            ):
                raise ValueError(
                    f"The embeddings deployment {unsupported_id!r} is mapped onto the chat completion deployment {supported_id!r}"
                )


def parse_compat_mapping(compat_mapping: Dict[str, str]):
    _validate_compat_mapping(compat_mapping)

    unknown: Dict[str, str] = {}
    chat: Dict[str, AdapterDeployment[ChatCompletionDeployment]] = {}
    embeddings: Dict[str, AdapterDeployment[EmbeddingsDeployment]] = {}

    for unsupported_id, supported_id in compat_mapping.items():
        if deployment := ChatCompletionDeployment.from_string(supported_id):
            chat[unsupported_id] = AdapterDeployment(
                upstream_deployment_id=unsupported_id,
                reference_deployment_id=deployment,
            )
        elif deployment := EmbeddingsDeployment.from_string(supported_id):
            embeddings[unsupported_id] = AdapterDeployment(
                upstream_deployment_id=unsupported_id,
                reference_deployment_id=deployment,
            )
        else:
            unknown[unsupported_id] = supported_id

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"Parsed compatibility mapping: {unknown=}, {chat=}, {embeddings=}"
        )

    if unknown:
        raise ValueError(
            f"None of the values in the following compatibility mapping corresponds to a "
            f"Bedrock deployment supported by the adapter: {json.dumps(unknown)}. "
            f"Remap the deployments to the supported Bedrock deployments to fix the error."
        )

    if chat or embeddings:
        log.warning(_compat_mapping_deprecation_warning(chat | embeddings))

    return chat, embeddings


def _compat_mapping_deprecation_warning(deployments: Dict[str, Any]) -> str:
    def create_model_entry(
        deployment: (
            AdapterDeployment[ChatCompletionDeployment]
            | AdapterDeployment[EmbeddingsDeployment]
        ),
    ) -> dict:
        is_chat = isinstance(
            deployment.reference_deployment_id, ChatCompletionDeployment
        )
        type = "chat" if is_chat else "embedding"
        endpoint = "chat/completions" if is_chat else "embeddings"
        compatible_id = deployment.reference_deployment_id.value
        extra = {"compatible_deployment_id": compatible_id}
        return {
            "type": type,
            "endpoint": f"$ADAPTER_ORIGIN/openai/deployments/{deployment.upstream_deployment_id}/{endpoint}",
            "upstreams": [{"extraData": extra}],
        }

    models, idx = {}, 1
    for deployment in sorted(deployments.values(), key=str):
        models[f"$DIAL_DEPLOYMENT_ID{idx}"] = create_model_entry(deployment)
        idx += 1
    config = {"models": models}

    return (
        f"{COMPAT_MAPPING_NAME} is deprecated in favour of per-upstream configuration in the DIAL Core config. "
        "You may remove the entries from the env variable one-by-one and amend configurations "
        f"for corresponding deployments in the DIAL Core config: {json.dumps(config)}"
    )
