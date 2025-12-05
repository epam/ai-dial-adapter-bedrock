import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import pytest
from aidial_sdk.exceptions import DeploymentNotFoundError

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.utils.adapter_deployments import (
    AdapterDeployments,
    resolve_upstream_deployment_id,
)


def _invalid_upstream_config_error(upstream_id: str, compat_id: str) -> str:
    return f"{compat_id!r} is declared as a deployment id that is compatible with {upstream_id!r} via upstreams[*].extraData.compatible_deployment_id field in the DIAL Core config. However, {compat_id!r} isn't one of the deployment ids supported by the adapter. Replace it with a supported deployment id to avoid this error."


def _invalid_compat_mapping_error(upstream_id: str, compat_id: str) -> str:
    return f"{compat_id!r} is declared as a deployment id that is compatible with {upstream_id!r} via COMPATIBILITY_MAPPING env variable. However, {compat_id!r} isn't one of the deployment ids supported by the adapter. Replace it with a supported deployment id to avoid this error."


def _invalid_upstream_deployment_id(deployment_id: str) -> str:
    return f"The deployment id {deployment_id!r} isn't one of the deployment ids supported by the adapter. Either replace it with a supported deployment id, or set upstreams[*].extraData.compatible_deployment_id field in the DIAL Core config equal to a supported deployment id that is compatible with {deployment_id!r}."


def _compat_mapping_deprecation_warning(
    unsupported_id: str, supported_id: str
) -> str:
    return (
        """
COMPATIBILITY_MAPPING env variable is deprecated in favour of per-upstream configuration in the DIAL Core config. You may remove the entries from the env variable one-by-one and amend configurations for corresponding deployments in the DIAL Core config: {"models": {"$DIAL_DEPLOYMENT_ID1": {"type": "chat", "endpoint": "$ADAPTER_ORIGIN/openai/deployments/$unsupported_id/chat/completions", "upstreams": [{"extraData": {"compatible_deployment_id": "$supported_id"}}]}}}
""".strip()
        .replace("$unsupported_id", unsupported_id)
        .replace("$supported_id", supported_id)
    )


_known_regions = ["us", "eu", "apac", "jp", "global"]


def _unknown_region_warning(deployment: str, region: str) -> str:
    return f"{deployment!r} has unexpected cross-region prefix {region!r} that doesn't match any of the supported prefixes: us, eu, apac, jp, global."


class Checker(Protocol):
    def check(self, deployments: AdapterDeployments): ...


@dataclass
class supported:
    deployment_id: ChatCompletionDeployment | EmbeddingsDeployment

    def check(self, deployments: AdapterDeployments):
        deployment_name = self.deployment_id.value
        if isinstance(self.deployment_id, ChatCompletionDeployment):
            deployment = deployments.chat.get(deployment_name)
        else:
            deployment = deployments.embeddings.get(deployment_name)

        assert deployment is not None
        assert deployment.upstream_deployment_id == deployment_name
        assert deployment.reference_deployment_id == self.deployment_id


@dataclass
class compat:
    deployment_id: str
    reference: ChatCompletionDeployment | EmbeddingsDeployment

    def check(self, deployments: AdapterDeployments):
        if isinstance(self.reference, ChatCompletionDeployment):
            deployment = deployments.chat.get(self.deployment_id)
        else:
            deployment = deployments.embeddings.get(self.deployment_id)

        assert deployment is not None
        assert deployment.upstream_deployment_id == self.deployment_id
        assert deployment.reference_deployment_id == self.reference


@dataclass
class TestCase:
    __test__ = False

    desc: str
    compat: Dict[str, str]

    error: str | None = None
    warning: str | None = None
    checks: List[Checker] = field(default_factory=list)


_CHAT_MODEL_1 = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU
_CHAT_MODEL_2 = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET

_EMBEDDING_MODEL = EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2

_outdated_mapping_warning_message = (
    "{deployment_id!r} deployment is already natively supported by the adapter, but it is also mapped to {supported_id!r} in the COMPATIBILITY_MAPPING env variable. "
    "To avoid this warning and ensure you retain all features of {deployment_id!r}, remove it from the mapping. "
    "Otherwise, you may lose features that exist in {deployment_id!r} but are missing in {supported_id!r}."
)

test_cases: List[TestCase] = [
    TestCase(
        desc="invalid compat",
        compat={"xxx": "yyy", "zzz": "ddd"},
        error='None of the values in the following compatibility mapping corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy", "zzz": "ddd"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="partially invalid compat",
        compat={
            "xxx": "yyy",
            "zzz": _CHAT_MODEL_1.value,
        },
        error='None of the values in the following compatibility mapping corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="compat chat+embeddings",
        compat={
            "xxx": _CHAT_MODEL_1.value,
            "yyy": _EMBEDDING_MODEL.value,
        },
        checks=[
            supported(_CHAT_MODEL_1),
            supported(_EMBEDDING_MODEL),
            compat("xxx", _CHAT_MODEL_1),
            compat("yyy", _EMBEDDING_MODEL),
        ],
    ),
    TestCase(
        desc="compat mismatching supported deployments #1",
        compat={
            _CHAT_MODEL_1.value: _EMBEDDING_MODEL.value,
        },
        error=(
            f"The chat completion deployment {_CHAT_MODEL_1.value!r} is mapped"
            f" onto the embeddings deployment {_EMBEDDING_MODEL.value!r}"
        ),
    ),
    TestCase(
        desc="compat mismatching supported deployments #2",
        compat={
            _EMBEDDING_MODEL.value: _CHAT_MODEL_1.value,
        },
        error=(
            f"The embeddings deployment {_EMBEDDING_MODEL.value!r} is mapped"
            f" onto the chat completion deployment {_CHAT_MODEL_1.value!r}"
        ),
    ),
    TestCase(
        desc="outdated compatibility mapping (original)",
        compat={_CHAT_MODEL_2.value: _CHAT_MODEL_1.value},
        warning=_outdated_mapping_warning_message.format(
            deployment_id=_CHAT_MODEL_2.value,
            supported_id=_CHAT_MODEL_1.value,
        ),
        checks=[
            supported(_CHAT_MODEL_1),
            compat(_CHAT_MODEL_2.value, _CHAT_MODEL_1),
        ],
    ),
    TestCase(
        desc="outdated compatibility mapping (regional)",
        compat={_CHAT_MODEL_2.US.value: _CHAT_MODEL_1.value},
        warning=_outdated_mapping_warning_message.format(
            deployment_id=_CHAT_MODEL_2.US.value,
            supported_id=_CHAT_MODEL_1.value,
        ),
        checks=[
            supported(_CHAT_MODEL_1),
            compat(_CHAT_MODEL_2.US.value, _CHAT_MODEL_1),
        ],
    ),
    TestCase(
        desc="compatibility mapping deprecation warning",
        compat={"xxx": _CHAT_MODEL_1.value},
        warning=_compat_mapping_deprecation_warning("xxx", _CHAT_MODEL_1.value),
        checks=[],
    ),
]


@pytest.mark.parametrize(
    "test_case", test_cases, ids=lambda t: t.desc.replace(" ", "_")
)
def test_static_compat_mapping(caplog, test_case: TestCase):
    with caplog.at_level(logging.WARNING):
        if test_case.error is not None:
            with pytest.raises(ValueError, match=test_case.error):
                AdapterDeployments.create(compat_mapping=test_case.compat)
        else:
            deployments = AdapterDeployments.create(
                compat_mapping=test_case.compat
            )
            for checker in test_case.checks:
                checker.check(deployments)

    if warn_message := test_case.warning:
        assert warn_message in caplog.text


def test_non_existing_upstream_deployment_from_request(caplog):
    with pytest.raises(DeploymentNotFoundError, match="Deployment not found"):
        resolve_upstream_deployment_id(
            ChatCompletionDeployment,
            upstream_deployment_id="xxx",
        )

    assert _invalid_upstream_deployment_id("xxx") in caplog.text


def test_existing_upstream_deployment_from_request_no_region():
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id=_CHAT_MODEL_1.value,
    )

    assert deployment.upstream_deployment_id == _CHAT_MODEL_1.value
    assert deployment.reference_deployment_id == _CHAT_MODEL_1


@pytest.mark.parametrize("region", _known_regions)
def test_existing_upstream_deployment_from_request_with_region(region: str):
    upstream = f"{region}.{_CHAT_MODEL_1.value}"
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment, upstream_deployment_id=upstream
    )

    assert deployment.upstream_deployment_id == upstream
    assert deployment.reference_deployment_id == _CHAT_MODEL_1


def test_existing_upstream_deployment_from_request_with_unknown_region(caplog):
    upstream = f"unknown_region.{_CHAT_MODEL_1.value}"
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment, upstream_deployment_id=upstream
    )

    assert deployment.upstream_deployment_id == upstream
    assert deployment.reference_deployment_id == _CHAT_MODEL_1

    assert _unknown_region_warning(upstream, "unknown_region") in caplog.text


def test_non_existing_upstream_deployment_from_compat_mapping(caplog):
    with pytest.raises(DeploymentNotFoundError, match="Deployment not found"):
        resolve_upstream_deployment_id(
            ChatCompletionDeployment,
            upstream_deployment_id="xxx",
            compat_mapping={"xxx": "yyy"},
        )

    assert _invalid_compat_mapping_error("xxx", "yyy") in caplog.text


def test_existing_upstream_deployment_from_compat_mapping():
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id="xxx",
        compat_mapping={"xxx": _CHAT_MODEL_1.value},
    )

    assert deployment.upstream_deployment_id == "xxx"
    assert deployment.reference_deployment_id == _CHAT_MODEL_1


@pytest.mark.parametrize("region", _known_regions)
def test_existing_upstream_deployment_from_compat_mapping_with_region(
    region: str,
):
    compat = f"{region}.{_CHAT_MODEL_1.value}"
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id="xxx",
        compat_mapping={"xxx": compat},
    )

    assert deployment.upstream_deployment_id == "xxx"
    assert deployment.reference_deployment_id == _CHAT_MODEL_1


def test_non_existing_upstream_deployment_from_upstream_config(caplog):
    with pytest.raises(DeploymentNotFoundError, match="Deployment not found"):
        resolve_upstream_deployment_id(
            ChatCompletionDeployment,
            upstream_deployment_id="xxx",
            compatible_id_from_upstream="yyy",
        )

    assert _invalid_upstream_config_error("xxx", "yyy") in caplog.text


def test_existing_upstream_deployment_from_upstream_config_no_compat_mapping():
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id="xxx",
        compatible_id_from_upstream=_CHAT_MODEL_1.value,
    )

    assert deployment.upstream_deployment_id == "xxx"
    assert deployment.reference_deployment_id == _CHAT_MODEL_1


@pytest.mark.parametrize("region", _known_regions)
def test_existing_upstream_deployment_from_upstream_config_no_compat_mapping_with_region(
    region: str,
):
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id="xxx",
        compatible_id_from_upstream=f"{region}.{_CHAT_MODEL_1.value}",
    )

    assert deployment.upstream_deployment_id == "xxx"
    assert deployment.reference_deployment_id == _CHAT_MODEL_1


def test_existing_upstream_deployment_from_upstream_config_no_compat_mapping_with_unknown_region(
    caplog,
):
    compat = f"unknown_region.{_CHAT_MODEL_1.value}"
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id="xxx",
        compatible_id_from_upstream=compat,
    )

    assert deployment.upstream_deployment_id == "xxx"
    assert deployment.reference_deployment_id == _CHAT_MODEL_1

    assert _unknown_region_warning(compat, "unknown_region") in caplog.text


def test_existing_upstream_deployment_from_upstream_config_with_compat_mapping():
    deployment = resolve_upstream_deployment_id(
        ChatCompletionDeployment,
        upstream_deployment_id="xxx",
        compat_mapping={"xxx": _CHAT_MODEL_2.value},
        compatible_id_from_upstream=_CHAT_MODEL_1.value,
    )

    assert deployment.upstream_deployment_id == "xxx"
    assert deployment.reference_deployment_id == _CHAT_MODEL_1
