"""
An attempt to approximate the tokenizer for Claude V3.

This tokenizer doesn't provide the precise token count, because Anthropic
doesn't provide the exact tokenization algorithm.

This tokenizer provides an *overestimation* of the request token count.
We need to be conservative, since the tokenizer is used in the prompt
truncation algorithm. So we are choosing to be unable to pack the request with tokens
as tightly as possible over making an additional chat completion request,
which is going to fail with a token overflow error.

1. For the text parts of request we use official Claude 2 tokenizer
with an overhead of 20%, since it's explicitly declared that for Claude 3
it only could be used as a very rough estimate:
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
from typing import (
    Awaitable,
    Callable,
    List,
    Literal,
    Tuple,
    Union,
    assert_never,
)

from anthropic._tokenizers import async_get_tokenizer
from anthropic._types import Base64FileInput
from anthropic.types import (
    ContentBlock,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock
from PIL import Image

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.llm.model.claude.v3.params import MessagesParams
from aidial_adapter_bedrock.utils.log_config import app_logger as log

_TEXT_OVERESTIMATION_FACTOR = 1.2

# Rough estimation
_PER_MESSAGE_TOKENS = 5


async def _tokenize_text(content: str) -> int:
    tokenizer = await async_get_tokenizer()
    tokens = len(tokenizer.encode(content).ids)
    return int(tokens * _TEXT_OVERESTIMATION_FACTOR)


def _get_image_size(image_data: Union[str, Base64FileInput]) -> Tuple[int, int]:
    try:
        if not isinstance(image_data, str):
            raise ValueError("Images as files aren't yet supported.")

        image_bytes = base64.b64decode(image_data)
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.size
    except Exception:
        log.error("Cannot compute image size, assuming 1000x1000")
        return 1000, 1000


async def _tokenize_image(source: Source) -> int:
    width, height = _get_image_size(source["data"])
    return int((width * height) / 750.0)


async def _tokenize_tool_use(id: str, input: object, name: str) -> int:
    return await _tokenize_text(f"{id} {name} {json.dumps(input)}")


async def _tokenize_tool_result(message: ToolResultBlockParam) -> int:
    tokens: int = await _tokenize_text(message["tool_use_id"])
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
                return await _tokenize_text(message["text"])
            case "image":
                return await _tokenize_image(message["source"])
            case "tool_use":
                return await _tokenize_tool_use(
                    message["id"], message["input"], message["name"]
                )
            case "tool_result":
                return await _tokenize_tool_result(message)
            case _:
                assert_never(message["type"])
    else:
        match message:
            case TextBlock():
                return await _tokenize_text(message.text)
            case ToolUseBlock():
                return await _tokenize_tool_use(
                    message.id, message.input, message.name
                )
            case _:
                assert_never(message)


async def _tokenize_message(message: MessageParam) -> int:
    tokens: int = 0

    content = message["content"]

    match content:
        case str():
            tokens += await _tokenize_text(content)
        case _:
            for item in content:
                tokens += await _tokenize_sub_message(item)

    return tokens


async def _tokenize_messages(messages: List[MessageParam]) -> int:
    tokens: int = 0
    for message in messages:
        tokens += await _tokenize_message(message) + _PER_MESSAGE_TOKENS
    return tokens


async def _tokenize_tool_param(tool: ToolParam) -> int:
    return await _tokenize_text(json.dumps(tool))


def _tokenize_tool_system_message(
    deployment_id: str,
    tool_choice: Literal["auto", "any", "tool"],
) -> int:
    match deployment_id:
        case ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET:
            return 294 if tool_choice == "auto" else 261
        case ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS:
            return 530 if tool_choice == "auto" else 281
        case ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET:
            return 159 if tool_choice == "auto" else 235
        case ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU:
            return 264 if tool_choice == "auto" else 340
        case _:
            raise RuntimeError(
                f"Expected Claude 3 model, but got: {deployment_id}"
            )


async def _tokenize(
    deployment_id: str, params: MessagesParams, messages: List[MessageParam]
) -> int:
    tokens: int = 0

    if tools := params["tools"]:
        if system := params["system"]:
            tokens += await _tokenize_text(system)

        if tool_choice := params["tool_choice"]:
            choice = tool_choice["type"]
        else:
            choice = "auto"

        tokens += _tokenize_tool_system_message(deployment_id, choice)

        for tool in tools:
            tokens += await _tokenize_tool_param(tool)

    tokens += await _tokenize_messages(messages)

    return tokens


def create_tokenizer(
    deployment_id: str, params: MessagesParams
) -> Callable[[List[MessageParam]], Awaitable[int]]:
    async def _tokenizer(messages: List[MessageParam]) -> int:
        return await _tokenize(deployment_id, params, messages)

    return _tokenizer
