# Adapter for Cohere models.
# See the documentation at https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from aidial_sdk.chat_completion import FinishReason, Message
from aidial_sdk.exceptions import InternalServerError
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import (
    Bedrock,
    Headers,
    ResponseWithInvocationMetricsMixin,
    usage_from_headers,
)
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    CueMapping,
    post_process_completion_stream,
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
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_COHERE
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)
from aidial_adapter_bedrock.utils.list_projection import ListProjection


class CohereGeneration(BaseModel):
    id: str
    index: int | None = None
    text: str
    finish_reason: str | None = None


class CohereResponse(ResponseWithInvocationMetricsMixin):
    id: str
    prompt: Optional[str]
    generations: List[CohereGeneration]

    def content(self) -> str:
        return self.generations[0].text


def _to_dial_finish_reason(reason: str) -> FinishReason | None:
    match reason:
        case "COMPLETE":
            return FinishReason.STOP
        case "MAX_TOKENS":
            return FinishReason.LENGTH
        case "ERROR_TOXIC":
            return FinishReason.CONTENT_FILTER
        case "ERROR":
            raise InternalServerError("The model returned an error.")
        case _:
            return None


def convert_params(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.temperature is not None:
        ret["temperature"] = params.temperature

    if params.max_tokens is not None:
        ret["max_tokens"] = params.max_tokens
    else:
        # Choosing reasonable default
        ret["max_tokens"] = DEFAULT_MAX_TOKENS_COHERE

    # NOTE: num_generations is supported

    return ret


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **params}


def _collect_finish_reasons(
    resp: CohereResponse, finish_reasons: FinishReasons
) -> None:
    for generation in resp.generations:
        if finish_reason := generation.finish_reason:
            index = generation.index or 0
            if reason := _to_dial_finish_reason(finish_reason):
                finish_reasons[index] = reason


def chunks_to_stream(
    chunks: AsyncIterator[dict],
) -> Tuple[AsyncIterator[str], CompletionState]:
    state = CompletionState()

    async def _gen():
        async for chunk in chunks:
            resp = CohereResponse.parse_obj(chunk)
            state.usage.accumulate(resp.usage_from_metrics())
            _collect_finish_reasons(resp, state.finish_reasons)
            yield resp.content()

    return _gen(), state


def response_to_stream(
    response_body: dict, response_headers: Headers
) -> Tuple[AsyncIterator[str], CompletionState]:
    state = CompletionState()

    resp = CohereResponse.parse_obj(response_body)
    state.usage.accumulate(usage_from_headers(response_headers))
    _collect_finish_reasons(resp, state.finish_reasons)

    async def _gen():
        yield resp.content()

    return _gen(), state


cohere_emulator = BasicChatEmulator(
    prelude_template=None,
    should_prefix_with_cue=lambda _, idx: idx > 0,
    should_add_invitation_cue=False,
    should_fallback_to_completion=False,
    cues=CueMapping(
        system="User:",
        human="User:",
        ai="Chatbot:",
    ),
    separator="\n",
)


def _preprocess_cohere_messages(
    messages: List[Message],
) -> ListProjection[Message]:
    ret = default_preprocess_messages(messages)

    # Cohere doesn't support empty messages,
    # so replace it with a single space.
    for msg in ret.raw_list:
        msg.content = msg.content or " "

    return ret


def create_adapter(client: Bedrock, model: str) -> ChatCompletionAdapter:
    return compose_decorators(
        preprocess_messages_decorator(_preprocess_cohere_messages),
        truncate_prompt_decorator(
            keep_message=keep_last_and_system_messages,
            partitioner=trivial_partitioner,
        ),
        replicator_decorator(),
        tools_emulator_decorator(default_tools_emulator),
    )(
        pseudo_chat_adapter(cohere_emulator)(
            CohereAdapter(client=client, model=model)
        )
    )


class CohereAdapter(TextCompletionAdapter):
    model: str
    client: Bedrock

    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream, state = chunks_to_stream(chunks)
        else:
            response, headers = await self.client.ainvoke_non_streaming(
                self.model, args
            )
            stream, state = response_to_stream(response, headers)

        stream = post_process_completion_stream(params, cohere_emulator, stream)

        async for content in stream:
            consumer.append_content(content)

        consumer.close_content(state.get_single_finish_reason())
        consumer.add_usage(state.usage)

    async def count_completion_tokens(self, string: str) -> int:
        return default_tokenize_string(string)

    async def count_prompt_tokens(
        self, params: ModelParameters, prompt: str
    ) -> int:
        return default_tokenize_string(prompt)
