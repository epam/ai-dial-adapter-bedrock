from typing import Any, AsyncIterator, Dict, Tuple

import anthropic
from aidial_sdk.chat_completion import FinishReason, Message, Role
from anthropic._tokenizers import async_get_tokenizer
from tokenizers import Tokenizer

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    ChatEmulator,
    CueMapping,
)
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    TextCompletionAdapter,
    default_preprocess_messages,
    keep_last_and_system_messages,
    trivial_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.pseudo_chat import pseudo_chat_adapter
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.decorator.tools_emulator import (
    tools_emulator_decorator,
)
from aidial_adapter_bedrock.llm.decorator.truncate_prompt import (
    truncate_prompt_decorator,
)
from aidial_adapter_bedrock.llm.model.completion_state import (
    CompletionState,
    FinishReasons,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.llm.tools.claude_emulator import (
    legacy_tools_emulator,
)
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)


def _to_dial_finish_reason(reason: str) -> FinishReason | None:
    match reason:
        case "stop_sequence":
            return FinishReason.STOP
        case "max_tokens":
            return FinishReason.LENGTH
        case _:
            return None


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


def _collect_finish_reasons(resp: dict, finish_reasons: FinishReasons) -> None:
    if finish_reason := resp.get("stop_reason"):
        if reason := _to_dial_finish_reason(finish_reason):
            finish_reasons[0] = reason


def chunks_to_stream(
    chunks: AsyncIterator[dict],
) -> Tuple[AsyncIterator[str], CompletionState]:
    state = CompletionState()

    async def _gen():
        async for chunk in chunks:
            _collect_finish_reasons(chunk, state.finish_reasons)
            yield chunk["completion"]

    return _gen(), state


def response_to_stream(
    response: dict,
) -> Tuple[AsyncIterator[str], CompletionState]:
    state = CompletionState()
    _collect_finish_reasons(response, state.finish_reasons)

    async def _gen():
        yield response["completion"]

    return _gen(), state


def get_anthropic_emulator(is_system_message_supported: bool) -> ChatEmulator:
    def add_cue(message: Message, idx: int) -> bool:
        if (
            idx == 0
            and message.role == Role.SYSTEM
            and is_system_message_supported
        ):
            return False
        return True

    return BasicChatEmulator(
        prelude_template=None,
        should_prefix_with_cue=add_cue,
        should_add_invitation_cue=True,
        should_fallback_to_completion=False,
        cues=CueMapping(
            system=anthropic.HUMAN_PROMPT.strip(),
            human=anthropic.HUMAN_PROMPT.strip(),
            ai=anthropic.AI_PROMPT.strip(),
        ),
        separator="\n\n",
    )


async def create_adapter(client: Bedrock, model: str) -> ChatCompletionAdapter:
    is_claude_v2_1 = (
        model == ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1.value
    )

    tools_emulator = (
        legacy_tools_emulator if is_claude_v2_1 else default_tools_emulator
    )

    chat_emulator = get_anthropic_emulator(
        is_system_message_supported=is_claude_v2_1
    )

    return compose_decorators(
        preprocess_messages_decorator(default_preprocess_messages),
        truncate_prompt_decorator(
            keep_message=keep_last_and_system_messages,
            partitioner=trivial_partitioner,
        ),
        replicator_decorator(),
        tools_emulator_decorator(tools_emulator),
    )(pseudo_chat_adapter(chat_emulator)(await Adapter.create(client, model)))


class Adapter(TextCompletionAdapter):
    model: str
    client: Bedrock
    tokenizer: Tokenizer

    @classmethod
    async def create(cls, client: Bedrock, model: str):
        return cls(
            client=client, model=model, tokenizer=await async_get_tokenizer()
        )

    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream, state = chunks_to_stream(chunks)
        else:
            response, _headers = await self.client.ainvoke_non_streaming(
                self.model, args
            )
            stream, state = response_to_stream(response)

        stream = stream_utils.lstrip(stream)

        completion = ""
        async for content in stream:
            completion += content
            consumer.append_content(content)

        consumer.close_content(state.get_single_finish_reason())
        consumer.add_usage(self._compute_usage(prompt, completion))

    def _compute_usage(self, prompt: str, completion: str) -> TokenUsage:
        batch = self.tokenizer.encode_batch([prompt, completion])

        return TokenUsage(
            prompt_tokens=len(batch[0].ids),
            completion_tokens=len(batch[1].ids),
        )

    async def count_prompt_tokens(
        self, params: ModelParameters, prompt: str
    ) -> int:
        return len(self.tokenizer.encode(prompt).ids)

    async def count_completion_tokens(self, string: str) -> int:
        return len(self.tokenizer.encode(string).ids)
