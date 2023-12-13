from typing import Any, AsyncIterator, Dict

import anthropic
from anthropic.tokenizer import count_tokens

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    ChatEmulator,
    CueMapping,
)
from aidial_adapter_bedrock.llm.chat_model import (
    PseudoChatModel,
    default_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage, SystemMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.llm.tools.claude import claude_v2_1_tools_emulator
from aidial_adapter_bedrock.llm.tools.emulator import default_tools_emulator


def compute_usage(prompt: str, completion: str) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=count_tokens(prompt),
        completion_tokens=count_tokens(completion),
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
    is_claude_v2_1: bool

    @classmethod
    def create(cls, client: Bedrock, model: str):
        is_claude_v2_1 = model == BedrockDeployment.ANTHROPIC_CLAUDE_V2_1

        chat_emulator = get_anthropic_emulator(
            is_system_message_supported=is_claude_v2_1
        )

        tools_emulator = (
            claude_v2_1_tools_emulator
            if is_claude_v2_1
            else default_tools_emulator
        )

        return cls(
            client=client,
            model=model,
            tokenize=count_tokens,
            chat_emulator=chat_emulator,
            tools_emulator=tools_emulator,
            partitioner=default_partitioner,
            is_claude_v2_1=is_claude_v2_1,
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

        consumer.add_usage(compute_usage(prompt, completion))
