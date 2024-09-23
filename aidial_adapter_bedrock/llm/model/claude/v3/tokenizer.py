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

import base64
import io
import json
import math
from typing import (
    Awaitable,
    Callable,
    List,
    Literal,
    Tuple,
    Union,
    assert_never,
)

from anthropic._types import Base64FileInput
from anthropic.types import ContentBlock, ImageBlockParam
from anthropic.types import MessageParam as ClaudeMessage
from anthropic.types import (
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock
from PIL import Image

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    Claude3Deployment,
)
from aidial_adapter_bedrock.llm.model.claude.v3.params import ClaudeParameters
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.utils.log_config import app_logger as log


def tokenize_text(text: str) -> int:
    return default_tokenize_string(text)


def _get_image_size(image_data: Union[str, Base64FileInput]) -> Tuple[int, int]:
    try:
        if not isinstance(image_data, str):
            raise ValueError("Images as files aren't yet supported.")

        image_bytes = base64.b64decode(image_data)
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.size
    except Exception:
        log.exception("Cannot compute image size, assuming 1000x1000")
        return 1000, 1000


async def _tokenize_image(source: Source) -> int:
    width, height = _get_image_size(source["data"])
    return math.ceil((width * height) / 750.0)


def _tokenize_tool_use(id: str, input: object, name: str) -> int:
    return tokenize_text(f"{id} {name} {json.dumps(input)}")


async def _tokenize_tool_result(message: ToolResultBlockParam) -> int:
    tokens: int = tokenize_text(message["tool_use_id"])
    if "content" in message:
        for sub_message in message["content"]:
            tokens += await _tokenize_sub_message(sub_message)
    return tokens


async def _tokenize_sub_message(
    message: Union[
        TextBlockParam,
        ImageBlockParam,
        ToolUseBlockParam,
        ToolResultBlockParam,
        ContentBlock,
    ]
) -> int:
    if isinstance(message, dict):
        match message["type"]:
            case "text":
                return tokenize_text(message["text"])
            case "image":
                return await _tokenize_image(message["source"])
            case "tool_use":
                return _tokenize_tool_use(
                    message["id"], message["input"], message["name"]
                )
            case "tool_result":
                return await _tokenize_tool_result(message)
            case _:
                assert_never(message["type"])
    else:
        match message:
            case TextBlock():
                return tokenize_text(message.text)
            case ToolUseBlock():
                return _tokenize_tool_use(
                    message.id, message.input, message.name
                )
            case _:
                assert_never(message)


async def _tokenize_message(message: ClaudeMessage) -> int:
    tokens: int = 0

    content = message["content"]

    match content:
        case str():
            tokens += tokenize_text(content)
        case _:
            for item in content:
                tokens += await _tokenize_sub_message(item)

    return tokens


async def _tokenize_messages(messages: List[ClaudeMessage]) -> int:
    # A rough estimation
    per_message_tokens = 5

    tokens: int = 0
    for message in messages:
        tokens += await _tokenize_message(message) + per_message_tokens
    return tokens


def _tokenize_tool_param(tool: ToolParam) -> int:
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
    params: ClaudeParameters,
    messages: List[ClaudeMessage],
) -> int:
    tokens: int = 0

    if system := params["system"]:
        tokens += tokenize_text(system)

    if tools := params["tools"]:
        if tool_choice := params["tool_choice"]:
            choice = tool_choice["type"]
        else:
            choice = "auto"

        tokens += _tokenize_tool_system_message(deployment, choice)

        for tool in tools:
            tokens += _tokenize_tool_param(tool)

    tokens += await _tokenize_messages(messages)

    return tokens


def create_tokenizer(
    deployment: Claude3Deployment, params: ClaudeParameters
) -> Callable[[List[ClaudeMessage]], Awaitable[int]]:
    async def _tokenizer(messages: List[ClaudeMessage]) -> int:
        return await _tokenize(deployment, params, messages)

    return _tokenizer
