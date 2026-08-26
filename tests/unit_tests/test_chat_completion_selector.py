import json
from dataclasses import dataclass

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.llm.converse.factory import ToolsSupport
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDocumentType,
    ConverseImageType,
)
from aidial_adapter_bedrock.llm.model.adapter import get_bedrock_adapter
from aidial_adapter_bedrock.upstream_config import CloudUpstreamConfig
from aidial_adapter_bedrock.utils.adapter_deployment import AdapterDeployment


@dataclass
class _FakeOriginalRequest:
    headers: dict[str, str]
    path_params: dict[str, str]


@dataclass
class _FakeRequest:
    api_key: str
    original_request: _FakeOriginalRequest


class TestChatCompletionSelector:
    async def test_get_model_passes_mantle_selector_to_adapter(
        self, monkeypatch
    ):
        captured = {}
        expected_model = object()

        async def _fake_get_bedrock_adapter(**kwargs):
            captured.update(kwargs)
            return expected_model

        monkeypatch.setattr(
            "aidial_adapter_bedrock.chat_completion.get_bedrock_adapter",
            _fake_get_bedrock_adapter,
        )

        request = _FakeRequest(
            api_key="dummy",
            original_request=_FakeOriginalRequest(
                headers={
                    "x-upstream-extra-data": json.dumps(
                        {
                            "region": "us-east-1",
                            "claude_client": "mantle",
                        }
                    )
                },
                path_params={
                    "deployment_id": D.ANTHROPIC_CLAUDE_V4_6_SONNET.value
                },
            ),
        )

        model = await BedrockChatCompletion()._get_model(request)  # type: ignore[arg-type]

        assert model is expected_model
        assert isinstance(captured["upstream_config"], CloudUpstreamConfig)
        assert captured["upstream_config"].claude_client == "mantle"
        assert captured["upstream_config"].region == "us-east-1"

    async def test_get_bedrock_adapter_uses_converse_for_converse_claude_client(
        self, monkeypatch
    ):
        captured = {}
        expected_adapter = object()

        class _FakeConverseAdapterFactory:
            def __init__(self, **kwargs):
                captured["factory_kwargs"] = kwargs

            async def create(self, **kwargs):
                captured["create_kwargs"] = kwargs
                return expected_adapter

        monkeypatch.setattr(
            "aidial_adapter_bedrock.llm.model.adapter.ConverseAdapterFactory",
            _FakeConverseAdapterFactory,
        )

        deployment = AdapterDeployment(
            upstream_deployment_id=D.ANTHROPIC_CLAUDE_V4_6_SONNET.value,
            reference_deployment_id=D.ANTHROPIC_CLAUDE_V4_6_SONNET,
        )

        adapter = await get_bedrock_adapter(
            deployment=deployment,
            api_key="dummy",
            upstream_config=CloudUpstreamConfig(
                region="us-east-1", claude_client="converse"
            ),
            request=None,
        )

        assert adapter is expected_adapter
        assert captured["factory_kwargs"]["deployment"] == deployment
        assert captured["factory_kwargs"]["api_key"] == "dummy"
        assert captured["create_kwargs"] == {
            "tools_support": ToolsSupport.ALWAYS,
            "supported_image_types": ConverseImageType.all(),
            "supported_document_types": ConverseDocumentType.all(),
        }
