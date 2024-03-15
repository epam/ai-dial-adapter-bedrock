from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, TypedDict

import anthropic
import httpx
from aidial_sdk.chat_completion import Message
from anthropic import NOT_GIVEN, AsyncAnthropic
from anthropic._tokenizers import async_get_tokenizer
from anthropic.lib.bedrock import AsyncAnthropicBedrock
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStreamEvent,
)
from tokenizers import Tokenizer

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    ChatEmulator,
    CueMapping,
)
from aidial_adapter_bedrock.llm.chat_model import (
    BaseChatModel,
    PseudoChatModel,
    default_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.llm.model.claude3.converters import (
    to_claude_messages,
    to_dial_finish_reason,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.llm.tools.claude_emulator import (
    claude_v2_1_tools_emulator,
)
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)


# NOTE: See https://docs.anthropic.com/claude/reference/complete_post
def convert_params(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.max_tokens is not None:
        ret["max_tokens_to_sample"] = params.max_tokens
    else:
        # The max tokens parameter is required for Anthropic models.
        # Choosing reasonable default.
        ret["max_tokens_to_sample"] = DEFAULT_MAX_TOKENS_ANTHROPIC

    if params.stop:
        ret["stop_sequences"] = params.stop

    if params.temperature is not None:
        ret["temperature"] = params.temperature

    if params.top_p is not None:
        ret["top_p"] = params.top_p

    return ret


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **params}


async def chunks_to_stream(
    chunks: AsyncIterator[dict],
) -> AsyncIterator[str]:
    async for chunk in chunks:
        yield chunk["completion"]


async def response_to_stream(response: dict) -> AsyncIterator[str]:
    yield response["completion"]


def get_anthropic_emulator(is_system_message_supported: bool) -> ChatEmulator:
    def add_cue(message: BaseMessage, idx: int) -> bool:
        if (
            idx == 0
            and isinstance(message, SystemMessage)
            and is_system_message_supported
        ):
            return False
        return True

    return BasicChatEmulator(
        prelude_template=None,
        add_cue=add_cue,
        add_invitation_cue=True,
        fallback_to_completion=False,
        cues=CueMapping(
            system=anthropic.HUMAN_PROMPT.strip(),
            human=anthropic.HUMAN_PROMPT.strip(),
            ai=anthropic.AI_PROMPT.strip(),
        ),
        separator="\n\n",
    )


class AnthropicAdapter(PseudoChatModel):
    client: Bedrock
    tokenizer: Tokenizer
    is_claude_v2_1: bool

    @classmethod
    async def create(cls, client: Bedrock, model: str):
        is_claude_v2_1 = model == BedrockDeployment.ANTHROPIC_CLAUDE_V2_1

        chat_emulator = get_anthropic_emulator(
            is_system_message_supported=is_claude_v2_1
        )

        tools_emulator = (
            claude_v2_1_tools_emulator
            if is_claude_v2_1
            else default_tools_emulator
        )

        tokenizer = await async_get_tokenizer()
        return cls(
            client=client,
            model=model,
            tokenize=lambda text: len(tokenizer.encode(text).ids),
            chat_emulator=chat_emulator,
            tools_emulator=tools_emulator,
            partitioner=default_partitioner,
            is_claude_v2_1=is_claude_v2_1,
            tokenizer=tokenizer,
        )

    async def _apredict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream = chunks_to_stream(chunks)
        else:
            response = await self.client.ainvoke_non_streaming(self.model, args)
            stream = response_to_stream(response)

        stream = stream_utils.lstrip(stream)

        completion = ""
        async for content in stream:
            completion += content
            consumer.append_content(content)
        consumer.close_content()

        consumer.add_usage(self._compute_usage(prompt, completion))

    def _compute_usage(self, prompt: str, completion: str) -> TokenUsage:
        batch = self.tokenizer.encode_batch([prompt, completion])

        return TokenUsage(
            prompt_tokens=len(batch[0].ids),
            completion_tokens=len(batch[1].ids),
        )


class UsageEventHandler(AsyncMessageStream):
    def __init__(
        self,
        *,
        cast_to: type[MessageStreamEvent],
        response: httpx.Response,
        client: AsyncAnthropic,
    ):
        super().__init__(cast_to=cast_to, response=response, client=client)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.stop_reason: str | None = None

    async def on_stream_event(self, event: MessageStreamEvent):
        if isinstance(event, MessageStartEvent):
            self.prompt_tokens = event.message.usage.input_tokens
        elif isinstance(event, MessageDeltaEvent):
            self.completion_tokens += event.usage.output_tokens
            self.stop_reason = event.delta.stop_reason


class CompletionParameters(TypedDict):
    messages: List[MessageParam]
    model: str
    max_tokens: int
    stop_sequences: Optional[List[str]]
    system: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]


class AnthropicChat(BaseChatModel):
    storage: Optional[FileStorage]

    async def achat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ):
        prompt, claude_messages = await to_claude_messages(
            messages, self.storage
        )
        completion_params = {
            "max_tokens": params.max_tokens or DEFAULT_MAX_TOKENS_ANTHROPIC,
            "stop_sequences": params.stop or NOT_GIVEN,
            "system": prompt or NOT_GIVEN,
            "temperature": params.temperature or NOT_GIVEN,
            "top_p": params.top_p or NOT_GIVEN,
        }
        if params.stream:
            await self.invoke_streaming(
                consumer, claude_messages, completion_params
            )
        else:
            await self.invoke(consumer, claude_messages, completion_params)

    async def invoke_streaming(
        self,
        consumer: Consumer,
        messages: List[MessageParam],
        params: dict[str, Any],
    ):
        client = AsyncAnthropicBedrock()
        async with client.messages.stream(
            messages=messages,
            model=self.model,
            event_handler=UsageEventHandler,
            **params,
        ) as stream:
            async for text in stream.text_stream:
                consumer.append_content(text)
            consumer.close_content(to_dial_finish_reason(stream.stop_reason))

            consumer.add_usage(
                TokenUsage(
                    prompt_tokens=stream.prompt_tokens,
                    completion_tokens=stream.completion_tokens,
                )
            )

    async def invoke(
        self,
        consumer: Consumer,
        messages: List[MessageParam],
        params: dict[str, Any],
    ):
        client = AsyncAnthropicBedrock()
        message = await client.messages.create(
            messages=messages, model=self.model, **params
        )
        prompt_tokens = 0
        completion_tokens = 0
        for content in message.content:
            usage = message.usage
            prompt_tokens = usage.input_tokens
            completion_tokens += usage.output_tokens
            consumer.append_content(content.text)
        consumer.close_content(to_dial_finish_reason(message.stop_reason))

        consumer.add_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @classmethod
    def create(cls, model: str, headers: Mapping[str, str]):
        storage: Optional[FileStorage] = create_file_storage(headers)
        return cls(
            model=model,
            storage=storage,
            tools_emulator=default_tools_emulator,
        )
