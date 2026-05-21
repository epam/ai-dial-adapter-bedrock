import pytest
from aidial_adapter_anthropic.adapter import ValidationError
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
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.caching import (
    get_response_headers_for_caching,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDocumentType,
    ConverseImageType,
)
from aidial_adapter_bedrock.upstream_config import CloudUpstreamConfig

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"


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
    function = Function(
        name="get_weather",
        description="Get the weather",
        parameters={"type": "object", "properties": {}},
    )
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
async def converse_adapter() -> ConverseAdapter:
    client = await Bedrock.acreate(CloudUpstreamConfig(region="us-east-1"))
    return ConverseAdapter(
        deployment="test",
        bedrock=client,
        support_tools=True,
        storage=None,
        supported_image_types=ConverseImageType.all(),
        supported_document_types=ConverseDocumentType.all(),
        ensure_non_empty_tool_descriptions=False,
    )


async def test_top_level_breakpoint_not_supported(
    converse_adapter: ConverseAdapter,
):
    params = ModelParameters(cache_breakpoint=CacheBreakpoint())
    with pytest.raises(
        ValidationError, match="Converse API does not support automatic caching"
    ):
        await converse_adapter.construct_converse_params(
            [_user("hello")], params
        )


def test_sets_headers_for_last_message_breakpoint():
    messages = [
        _user("first", cache_breakpoint={"ttl": "1h"}),
        _user("second"),
        _user("third", cache_breakpoint={"ttl": "5m"}),
    ]

    headers = get_response_headers_for_caching(messages, [])

    assert headers == {
        _DIAL_CACHE_BREAKPOINT_PATH: "prefix.body.messages[2]",
        _DIAL_CACHE_EXPIRE_AT: "4600",
    }


def test_does_not_set_headers_without_breakpoints():
    headers = get_response_headers_for_caching(
        [_user("first"), _user("second")], []
    )

    assert headers is None


def test_sets_headers_for_system_message_breakpoint():
    messages = [
        _sys("be helpful", cache_breakpoint={"ttl": "5m"}),
        _user("hello"),
    ]

    headers = get_response_headers_for_caching(messages, [])

    assert headers == {
        _DIAL_CACHE_BREAKPOINT_PATH: "prefix.body.messages[0]",
        _DIAL_CACHE_EXPIRE_AT: "1300",
    }


def test_sets_headers_for_tool_breakpoint():
    tools = [_tool(cache_breakpoint={})]

    headers = get_response_headers_for_caching(
        [_user("What's the weather?")], tools
    )

    assert headers == {
        _DIAL_CACHE_BREAKPOINT_PATH: "prefix.body.tools[0]",
        _DIAL_CACHE_EXPIRE_AT: "1300",
    }


def test_uses_default_ttl_for_invalid_breakpoint_ttl():
    messages = [
        _user("first"),
        _user("second", cache_breakpoint={"ttl": "invalid"}),
    ]

    headers = get_response_headers_for_caching(messages, [])

    assert headers == {
        _DIAL_CACHE_BREAKPOINT_PATH: "prefix.body.messages[1]",
        _DIAL_CACHE_EXPIRE_AT: "1300",
    }


def test_prefers_message_path_over_tool_breakpoint():
    tools = [_tool(cache_breakpoint={"ttl": "1h"})]
    messages = [_user("first", cache_breakpoint={"ttl": "5m"})]

    headers = get_response_headers_for_caching(messages, tools)

    assert headers == {
        _DIAL_CACHE_BREAKPOINT_PATH: "prefix.body.messages[0]",
        _DIAL_CACHE_EXPIRE_AT: "4600",
    }


def test_sets_headers_for_last_tool_breakpoint():
    tools = [
        _tool(),
        _tool(cache_breakpoint={"ttl": "5m"}),
    ]

    headers = get_response_headers_for_caching(
        [_user("What's the weather?")], tools
    )

    assert headers == {
        _DIAL_CACHE_BREAKPOINT_PATH: "prefix.body.tools[1]",
        _DIAL_CACHE_EXPIRE_AT: "1300",
    }
