from unittest.mock import MagicMock

import pytest
from aidial_adapter_anthropic.adapter import (
    ChatCompletionAdapter,
    ValidationError,
)
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_sdk.chat_completion import (
    Function,
    MessageCustomFields,
    Role,
    ToolCustomFields,
)
from aidial_sdk.chat_completion.request import (
    CacheBreakpoint,
    Message,
    Tool,
)

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.converse import caching as caching_module
from aidial_adapter_bedrock.llm.converse.caching import get_cache_info
from aidial_adapter_bedrock.llm.converse.factory import ConverseAdapterFactory
from aidial_adapter_bedrock.upstream_config import CloudUpstreamConfig


def _message(
    role: Role, content: str, cache_breakpoint: dict | None
) -> Message:
    custom_fields = None
    if cache_breakpoint is not None:
        custom_fields = MessageCustomFields(
            cache_breakpoint=CacheBreakpoint(**cache_breakpoint)
        )
    return Message(role=role, content=content, custom_fields=custom_fields)


def _user(content: str, *, cache_breakpoint: dict | None = None) -> Message:
    return _message(Role.USER, content, cache_breakpoint=cache_breakpoint)


def _sys(content: str, *, cache_breakpoint: dict | None = None) -> Message:
    return _message(Role.SYSTEM, content, cache_breakpoint=cache_breakpoint)


def _tool(*, cache_breakpoint: dict | None = None) -> Tool:
    function = Function(name="get_current_time")
    custom_fields = None
    if cache_breakpoint is not None:
        custom_fields = ToolCustomFields(
            cache_breakpoint=CacheBreakpoint(**cache_breakpoint)
        )
    return Tool(type="function", function=function, custom_fields=custom_fields)


@pytest.fixture(autouse=True)
def mock_current_time_1000s(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(caching_module.time, "time", lambda: 1000)


@pytest.fixture
async def adapter() -> ChatCompletionAdapter:
    async def get_client() -> Bedrock:
        return await Bedrock.acreate(CloudUpstreamConfig(region="test-region"))

    return await ConverseAdapterFactory(
        deployment="test-deployment-id",
        api_key="test-api-key",
        get_client=get_client,
    ).create()


async def test_top_level_breakpoint_not_supported(
    adapter: ChatCompletionAdapter,
):
    params = ModelParameters(cache_breakpoint=CacheBreakpoint())
    with pytest.raises(
        ValidationError, match="Converse API does not support automatic caching"
    ):
        await adapter.chat(MagicMock(), params, [_user("hello")])


def test_sets_headers_for_last_message_breakpoint():
    messages = [
        _user("first", cache_breakpoint={"ttl": "1h"}),
        _user("second"),
        _user("third", cache_breakpoint={"ttl": "5m"}),
    ]

    info = get_cache_info(messages, [])
    assert info is not None

    assert info.breakpoint_path.path == "prefix.body.messages[2]"
    assert info.expire_at == "4600"


def test_does_not_set_headers_without_breakpoints():
    info = get_cache_info([_user("first"), _user("second")], [])

    assert info is None


def test_sets_headers_for_system_message_breakpoint():
    messages = [
        _sys("be helpful", cache_breakpoint={"ttl": "5m"}),
        _user("hello"),
    ]

    info = get_cache_info(messages, [])
    assert info is not None

    assert info.breakpoint_path.path == "prefix.body.messages[0]"
    assert info.expire_at == "1300"


def test_sets_headers_for_tool_breakpoint():
    tools = [_tool(cache_breakpoint={})]

    info = get_cache_info([_user("What's the weather?")], tools)
    assert info is not None

    assert info.breakpoint_path.path == "prefix.body.tools[0]"
    assert info.expire_at == "1300"


def test_uses_default_ttl_for_invalid_breakpoint_ttl():
    messages = [
        _user("first"),
        _user("second", cache_breakpoint={"ttl": "invalid"}),
    ]

    info = get_cache_info(messages, [])
    assert info is not None

    assert info.breakpoint_path.path == "prefix.body.messages[1]"
    assert info.expire_at == "1300"


def test_prefers_message_path_over_tool_breakpoint():
    tools = [_tool(cache_breakpoint={"ttl": "1h"})]
    messages = [_user("first", cache_breakpoint={"ttl": "5m"})]

    info = get_cache_info(messages, tools)
    assert info is not None

    assert info.breakpoint_path.path == "prefix.body.messages[0]"
    assert info.expire_at == "4600"


def test_sets_headers_for_last_tool_breakpoint():
    tools = [
        _tool(),
        _tool(cache_breakpoint={"ttl": "5m"}),
    ]

    info = get_cache_info([_user("What's the weather?")], tools)
    assert info is not None

    assert info.breakpoint_path.path == "prefix.body.tools[1]"
    assert info.expire_at == "1300"
