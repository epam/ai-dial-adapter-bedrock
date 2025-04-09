from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import ContextManager, Optional, Protocol, Self

from aidial_sdk.chat_completion import (
    Attachment,
    Choice,
    FinishReason,
    FunctionCall,
    Response,
    ToolCall,
)

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.lazy_stage import LazyStage
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


class ToolUseMessage(Protocol):
    def append_arguments(self, arguments: str) -> Self: ...


class Consumer(ContextManager, ABC):
    @abstractmethod
    def fork(self) -> Consumer: ...

    @property
    @abstractmethod
    def choice(self) -> Choice: ...

    @abstractmethod
    def close_content(self, finish_reason: FinishReason | None = None): ...

    @abstractmethod
    def append_content(self, content: str): ...

    @abstractmethod
    def add_attachment(self, attachment: Attachment): ...

    @abstractmethod
    def add_usage(self, usage: TokenUsage): ...

    @abstractmethod
    def set_discarded_messages(
        self, discarded_messages: Optional[DiscardedMessages]
    ): ...

    @abstractmethod
    def get_discarded_messages(self) -> Optional[DiscardedMessages]: ...

    @abstractmethod
    def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage: ...

    @abstractmethod
    def create_function_call(self, call: FunctionCall) -> ToolUseMessage: ...

    @property
    @abstractmethod
    def has_function_call(self) -> bool: ...

    def create_stage(self, title: str) -> LazyStage:
        # NOTE: eta conversion to `factory = self.choice.create_stage`
        # is invalid, since `self.choice` must be created lazily.
        def factory(content: str):
            return self.choice.create_stage(content)

        return LazyStage(factory, title)


class ChoiceConsumer(Consumer):
    response: Response

    usage: Optional[TokenUsage]
    discarded_messages: Optional[DiscardedMessages]

    _root: Optional[Consumer]
    _choice: Optional[Choice]

    def __init__(self, response: Response, root: Optional[Consumer] = None):
        self.response = response

        self.usage = None
        self.discarded_messages = None

        self._choice = None
        self._root = root

    def fork(self) -> Consumer:
        return ChoiceConsumer(self.response, self._root or self)

    @property
    def choice(self) -> Choice:
        if self._choice is None:
            choice = self._choice = self.response.create_choice()
            # Delay opening a choice to the very last moment
            # so as to give opportunity for exceptions to bubble up to
            # the level of HTTP response (instead of error objects in a stream).
            choice.open()
            return choice
        else:
            return self._choice

    def __enter__(self) -> ChoiceConsumer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if exc is None and self._choice is not None:
            self._choice.close()

        if self._root is None:
            if self.usage is not None:
                self.response.set_usage(
                    self.usage.prompt_tokens,
                    self.usage.completion_tokens,
                )

            if self.discarded_messages is not None:
                self.response.set_discarded_messages(self.discarded_messages)

        return False

    def close_content(self, finish_reason: FinishReason | None = None):
        # Choice.close(finish_reason: Optional[FinishReason]) can be called only once
        # Currently, there's no other way to explicitly set the finish reason
        self.choice._last_finish_reason = finish_reason

    def append_content(self, content: str):
        self.choice.append_content(content)

    def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(attachment)

    def add_usage(self, usage: TokenUsage):
        if self._root:
            self._root.add_usage(usage)
        else:
            self.usage = (self.usage or TokenUsage()).accumulate(usage)

    def set_discarded_messages(
        self, discarded_messages: Optional[DiscardedMessages]
    ):
        if self._root:
            self._root.set_discarded_messages(discarded_messages)
        else:
            self.discarded_messages = discarded_messages

    def get_discarded_messages(self) -> Optional[DiscardedMessages]:
        if self._root:
            return self._root.get_discarded_messages()
        else:
            return self.discarded_messages

    def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        return self.choice.create_function_tool_call(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )

    def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        return self.choice.create_function_call(
            name=call.name,
            arguments=call.arguments,
        )

    @property
    def has_function_call(self) -> bool:
        return self._choice is not None and self._choice.has_function_call


class ConsumerDecorator(Consumer):
    consumer: Consumer

    def __init__(self, consumer: Consumer):
        self.consumer = consumer

    def __enter__(self) -> Consumer:
        return self.consumer.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.consumer.__exit__(exc_type, exc, traceback)

    def fork(self) -> Consumer:
        return self.consumer.fork()

    @property
    def choice(self) -> Choice:
        return self.consumer.choice

    def close_content(self, finish_reason: FinishReason | None = None):
        self.consumer.close_content(finish_reason)

    def append_content(self, content: str):
        self.consumer.append_content(content)

    def add_attachment(self, attachment: Attachment):
        self.consumer.add_attachment(attachment)

    def add_usage(self, usage: TokenUsage):
        self.consumer.add_usage(usage)

    def set_discarded_messages(
        self, discarded_messages: Optional[DiscardedMessages]
    ):
        self.consumer.set_discarded_messages(discarded_messages)

    def get_discarded_messages(self) -> Optional[DiscardedMessages]:
        return self.consumer.get_discarded_messages()

    def create_function_tool_call(self, call: ToolCall) -> ToolUseMessage:
        return self.consumer.create_function_tool_call(call)

    def create_function_call(self, call: FunctionCall) -> ToolUseMessage:
        return self.consumer.create_function_call(call)

    @property
    def has_function_call(self) -> bool:
        return self.consumer.has_function_call
