from dataclasses import dataclass

import pytest

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    CloudUpstreamConfig,
)


@dataclass
class _DummyClient:
    kind: str
    kwargs: dict


class TestCreateAnthropicClient:
    async def test_api_key_path_uses_async_anthropic(self, monkeypatch):
        create_anthropic_client.clear()

        def _fake_async_anthropic(**kwargs):
            return _DummyClient("api-key", kwargs)

        monkeypatch.setattr(
            "aidial_adapter_bedrock.bedrock.AsyncAnthropic",
            _fake_async_anthropic,
        )

        client = await create_anthropic_client(
            ApiKeyUpstreamConfig(api_key="test-key")
        )

        assert isinstance(client, _DummyClient)
        assert client.kind == "api-key"
        assert client.kwargs["api_key"] == "test-key"

    async def test_cloud_path_uses_legacy_client_by_default(self, monkeypatch):
        create_anthropic_client.clear()

        def _fake_legacy_client(**kwargs):
            return _DummyClient("legacy", kwargs)

        monkeypatch.setattr(
            "aidial_adapter_bedrock.bedrock.AsyncAnthropicBedrock",
            _fake_legacy_client,
        )

        client = await create_anthropic_client(
            CloudUpstreamConfig(region="us-east-1", claude_client="legacy")
        )

        assert isinstance(client, _DummyClient)
        assert client.kind == "legacy"
        assert client.kwargs["aws_region"] == "us-east-1"

    async def test_cloud_path_uses_mantle_client_when_selected(
        self, monkeypatch
    ):
        create_anthropic_client.clear()

        def _fake_mantle_client(**kwargs):
            return _DummyClient("mantle", kwargs)

        monkeypatch.setattr(
            "aidial_adapter_bedrock.bedrock.AsyncAnthropicBedrockMantle",
            _fake_mantle_client,
        )

        client = await create_anthropic_client(
            CloudUpstreamConfig(region="us-east-1", claude_client="mantle")
        )

        assert isinstance(client, _DummyClient)
        assert client.kind == "mantle"
        assert client.kwargs["aws_region"] == "us-east-1"

    async def test_cloud_path_rejects_boto_client(self):
        create_anthropic_client.clear()

        with pytest.raises(ValueError) as exc_info:
            await create_anthropic_client(
                CloudUpstreamConfig(
                    region="us-east-1", claude_client="converse"
                )
            )

        assert (
            str(exc_info.value)
            == "Claude client `converse` isn't supported for Anthropic API requests"
        )

    async def test_cache_key_differs_between_legacy_and_mantle(
        self, monkeypatch
    ):
        create_anthropic_client.clear()

        calls = {"legacy": 0, "mantle": 0}

        def _fake_legacy_client(**kwargs):
            calls["legacy"] += 1
            return _DummyClient("legacy", kwargs)

        def _fake_mantle_client(**kwargs):
            calls["mantle"] += 1
            return _DummyClient("mantle", kwargs)

        monkeypatch.setattr(
            "aidial_adapter_bedrock.bedrock.AsyncAnthropicBedrock",
            _fake_legacy_client,
        )
        monkeypatch.setattr(
            "aidial_adapter_bedrock.bedrock.AsyncAnthropicBedrockMantle",
            _fake_mantle_client,
        )

        legacy_1 = await create_anthropic_client(
            CloudUpstreamConfig(region="us-east-1", claude_client="legacy")
        )
        legacy_2 = await create_anthropic_client(
            CloudUpstreamConfig(region="us-east-1", claude_client="legacy")
        )
        mantle_1 = await create_anthropic_client(
            CloudUpstreamConfig(region="us-east-1", claude_client="mantle")
        )
        mantle_2 = await create_anthropic_client(
            CloudUpstreamConfig(region="us-east-1", claude_client="mantle")
        )

        assert legacy_1 is legacy_2
        assert mantle_1 is mantle_2
        assert legacy_1 is not mantle_1
        assert calls == {"legacy": 1, "mantle": 1}
