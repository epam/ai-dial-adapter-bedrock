import re
from typing import List, Optional

from langchain_core.callbacks import Callbacks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import AzureChatOpenAI

from tests.conftest import DEFAULT_API_VERSION
from tests.utils.callback import CallbackWithNewLines


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def user(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def sanitize_test_name(name: str) -> str:
    name2 = "".join(c if c.isalnum() else "_" for c in name.lower())
    return re.sub("_+", "_", name2)


async def run_model(
    model: BaseChatModel,
    messages: List[BaseMessage],
    streaming: bool,
    stop: Optional[List[str]],
) -> str:
    llm_result = await model.agenerate([messages], stop=stop)

    actual_usage = (
        llm_result.llm_output.get("token_usage", None)
        if llm_result.llm_output
        else None
    )

    # Usage is missing when and only where streaming is enabled
    assert (actual_usage in [None, {}]) == streaming

    return llm_result.generations[0][-1].text


def create_chat_model(
    base_url: str,
    model_id: str,
    streaming: bool,
    max_tokens: Optional[int],
) -> BaseChatModel:
    callbacks: Callbacks = [CallbackWithNewLines()]
    return AzureChatOpenAI(
        azure_endpoint=base_url,
        azure_deployment=model_id,
        callbacks=callbacks,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
        verbose=True,
        streaming=streaming,
        temperature=0,
        max_retries=0,
        max_tokens=max_tokens,
        request_timeout=10,  # type: ignore
    )
