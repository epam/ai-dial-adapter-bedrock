from typing import Callable, List

from aidial_adapter_anthropic.dial_api.request import ModelParameters
from aidial_adapter_anthropic.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_anthropic.llm.consumer import Consumer
from aidial_sdk.chat_completion import Message
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.chat_model import TextCompletionAdapter
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def pseudo_chat_adapter(
    chat_emulator: ChatEmulator,
) -> Callable[[TextCompletionAdapter], ChatCompletionAdapter]:
    return lambda adapter: PseudoChatAdapter(
        chat_emulator=chat_emulator, adapter=adapter
    )


class PseudoChatAdapter(ChatCompletionAdapter):
    chat_emulator: ChatEmulator
    adapter: TextCompletionAdapter

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        prompt, stop_sequences = self.chat_emulator.display(messages)
        params = params.add_stop_sequences(stop_sequences)

        log.debug(f"model parameters: {params.json(exclude_none=True)}")
        log.debug(f"prompt: {prompt!r}")

        await self.adapter.predict(consumer, params, prompt)

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        return await self.adapter.count_prompt_tokens(
            params, self.chat_emulator.display(messages)[0]
        )

    async def count_completion_tokens(self, string: str) -> int:
        return await self.adapter.count_completion_tokens(string)

    async def configuration(self) -> type[BaseModel]:
        raise NotImplementedError()

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> List[int] | None:
        raise NotImplementedError()
