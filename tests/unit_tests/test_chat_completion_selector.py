import json
from dataclasses import dataclass

from aidial_adapter_bedrock.chat_completion import BedrockChatCompletion
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.upstream_config import CloudUpstreamConfig


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
