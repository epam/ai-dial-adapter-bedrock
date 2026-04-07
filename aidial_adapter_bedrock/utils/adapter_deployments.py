from __future__ import annotations

from typing import TypeVar

from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from aidial_sdk.exceptions import DeploymentNotFoundError
from pydantic import BaseModel

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.upstream_config import get_compatible_model_id
from aidial_adapter_bedrock.utils.adapter_deployment import (
    AdapterDeployment,
    ReadableStrEnumT,
)
from aidial_adapter_bedrock.utils.compatibility_mapping import (
    COMPAT_MAPPING_NAME,
    COMPATIBILITY_MAPPING,
    parse_compat_mapping,
)
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_UPSTREAM_CONFIG_PATH = (
    "upstreams[*].extraData.compatible_model_id field in the DIAL Core config"
)

_T = TypeVar("_T", bound=ReadableStrEnumT)


def resolve_upstream_deployment_id_from_request(
    cls: type[_T], request: FromRequestDeploymentMixin
) -> AdapterDeployment[_T]:
    deployment_id = request.original_request.path_params["deployment_id"]
    return resolve_upstream_deployment_id(
        cls,
        upstream_deployment_id=deployment_id,
        compat_mapping=COMPATIBILITY_MAPPING,
        compatible_id_from_upstream=get_compatible_model_id(request),
    )


def resolve_upstream_deployment_id(
    cls: type[_T],
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

    compatible_model_id: _T | None = cls.from_string(compatible_id)

    if compatible_model_id is None:
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
                f"{compatible_id!r} is declared as a deployment id that is compatible with {upstream_deployment_id!r} via {COMPAT_MAPPING_NAME}. "
                f"However, {compatible_id!r} isn't one of the deployment ids supported by the adapter. "
                f"Replace it with a supported deployment id to avoid this error."
            )

        # Note that the diagnostic message isn't returned to the user in the response,
        # since it's intended for the DIAL admins, not users who cannot
        # do anything if the DIAL Core is misconfigured.
        # We return 404 error to the client though.
        # It's debatable, but returning 500 would have been too obscure
        # for the client.
        log.error(msg)
        raise DeploymentNotFoundError("Deployment not found")

    return AdapterDeployment[_T](
        upstream_deployment_id=upstream_deployment_id,
        reference_deployment_id=compatible_model_id,
    )


AdapterChatCompletionDeployment = AdapterDeployment[ChatCompletionDeployment]
AdapterEmbeddingsDeployment = AdapterDeployment[EmbeddingsDeployment]


class AdapterDeployments(BaseModel):
    chat: dict[str, AdapterChatCompletionDeployment]
    embeddings: dict[str, AdapterEmbeddingsDeployment]

    @classmethod
    def create(cls, *, compat_mapping: dict[str, str]) -> AdapterDeployments:
        chat, embeddings = parse_compat_mapping(compat_mapping)
        ret = cls(chat=chat, embeddings=embeddings)
        return ret._enrich_with_supported_deployments()

    def _enrich_with_supported_deployments(self) -> AdapterDeployments:
        chat = self.chat.copy()
        for deployment in ChatCompletionDeployment:
            for variant in deployment.variants:
                if variant not in self.chat:
                    chat[variant] = AdapterDeployment(
                        upstream_deployment_id=variant,
                        reference_deployment_id=deployment,
                    )

        embeddings = self.embeddings.copy()
        for deployment in EmbeddingsDeployment:
            variant = deployment.value
            if variant not in self.embeddings:
                embeddings[variant] = AdapterDeployment(
                    upstream_deployment_id=variant,
                    reference_deployment_id=deployment,
                )

        return AdapterDeployments(chat=chat, embeddings=embeddings)


def get_static_deployments() -> AdapterDeployments:
    return AdapterDeployments.create(compat_mapping=COMPATIBILITY_MAPPING)
