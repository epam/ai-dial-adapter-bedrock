"""
An attempt to approximate the tokenizer for Claude V3.

This tokenizer doesn't provide the precise token count,
because Anthropic doesn't provide the exact tokenization algorithm.

This tokenizer provides an *overestimation* of the request token count.
We need to be conservative, since the tokenizer is used in the prompt
truncation algorithm. So we are choosing to be unable to pack the request with tokens
as tightly as possible over making an additional chat completion request,
which is going to fail with a token overflow error.

1. For the text parts of request we count every byte in their UTF-8 encoding.
Note that the official Claude 2 tokenizer couldn't be used
for anything more than a very rough estimate:
https://github.com/anthropics/anthropic-sdk-python/blob/246a2978694b584429d4bbd5b44245ff8eac2ac2/src/anthropic/_client.py#L270-L283

2. For the image parts we use the official approximation:
> tokens = (width px * height px)/750
https://docs.anthropic.com/en/docs/build-with-claude/vision#calculate-image-costs

3. For the tool usage we use the official approximation:
https://docs.anthropic.com/en/docs/build-with-claude/tool-use#pricing
    a. tool-related components of the request are serialized to strings and tokenized as such,
    b. the hidden tool-enabling system prompt is accounted as per the documentation.
"""

import io
import json
import math
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    Tuple,
    assert_never,
    cast,
    get_args,
)

from PIL import Image

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    Claude3Deployment,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseContentPart,
    ConverseImagePartConfig,
    ConverseMessage,
    ConverseParams,
    ConverseToolConfig,
    ConverseToolResultConfig,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.utils.log_config import app_logger as log


def tokenize_text(text: str) -> int:
    return default_tokenize_string(text)


def _get_image_size(
    image_data: bytes, format: Literal["png", "jpeg", "gif", "webp"]
) -> Tuple[int, int]:
    try:
        with Image.open(io.BytesIO(image_data), formats=[format]) as img:
            return img.size
    except Exception:
        log.exception("Cannot compute image size, assuming 1000x1000")
        return 1000, 1000


async def _tokenize_image(part: ConverseImagePartConfig) -> int:
    width, height = _get_image_size(part["source"]["bytes"], part["format"])
    return math.ceil((width * height) / 750.0)


def _tokenize_tool_use(id: str, input: object, name: str) -> int:
    return tokenize_text(f"{id} {name} {json.dumps(input)}")


async def _tokenize_tool_result(tool_result: ConverseToolResultConfig) -> int:
    tokens: int = tokenize_text(tool_result["toolUseId"])
    if "content" in tool_result:
        for sub_message in tool_result["content"]:
            tokens += await _tokenize_sub_message(sub_message)
    return tokens


async def _tokenize_sub_message(message: ConverseContentPart) -> int:
    if text := message.get("text"):
        return tokenize_text(text)

    if json_content := message.get("json"):
        return tokenize_text(json.dumps(json_content))

    elif image := message.get("image"):
        return await _tokenize_image(image)

    elif tool_use := message.get("toolUse"):
        return _tokenize_tool_use(
            tool_use["toolUseId"], tool_use["input"], tool_use["name"]
        )
    elif tool_result := message.get("toolResult"):
        return await _tokenize_tool_result(tool_result)
    else:
        raise RuntimeError("Unexpected message content")


async def _tokenize_message(message: ConverseMessage) -> int:
    tokens: int = 0

    content = message["content"]
    for item in content:
        tokens += await _tokenize_sub_message(item)
    return tokens


async def _tokenize_messages(messages: List[ConverseMessage]) -> int:
    # A rough estimation
    per_message_tokens = 5

    tokens: int = 0
    for message in messages:
        tokens += await _tokenize_message(message) + per_message_tokens
    return tokens


def _tokenize_tool_param(tool: ConverseToolConfig) -> int:
    return tokenize_text(json.dumps(tool))


def _tokenize_tool_system_message(
    deployment: Claude3Deployment,
    tool_choice: Literal["auto", "any", "tool"],
) -> int:
    match deployment:
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_EU
        ):
            return 294 if tool_choice == "auto" else 261
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US
        ):
            return 530 if tool_choice == "auto" else 281
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_EU
        ):
            return 159 if tool_choice == "auto" else 235
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU
        ):
            return 264 if tool_choice == "auto" else 340
        case _:
            assert_never(deployment)


async def _tokenize(
    deployment: Claude3Deployment,
    params: ConverseParams,
    messages: List[ConverseMessage],
) -> int:
    tokens: int = 0

    if system := params.system:
        tokens += tokenize_text(system[0]["text"])

    if tools := params.toolConfig:
        if tool_choice := tools["toolChoice"]:
            tool_choice = next(iter(tool_choice.keys()))
        else:
            tool_choice = "auto"

        tokens += _tokenize_tool_system_message(deployment, tool_choice)

        for tool in tools["tools"]:
            tokens += _tokenize_tool_param(tool)

    tokens += await _tokenize_messages(messages)

    return tokens


def create_tokenizer(
    deployment: str, params: ConverseParams
) -> Callable[[List[Tuple[ConverseMessage, Any]]], Awaitable[int]]:
    if deployment not in get_args(Claude3Deployment):
        raise ValueError(f"Unsupported deployment: {deployment}")

    deployment = cast(Claude3Deployment, deployment)

    async def _tokenizer(messages) -> int:
        return await _tokenize(deployment, params, [msg for msg, _ in messages])

    return _tokenizer
