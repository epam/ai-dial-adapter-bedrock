import re
from typing import List

from langchain.callbacks.base import Callbacks
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

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
    model: BaseChatModel, messages: List[BaseMessage], streaming: bool
) -> str:
    llm_result = await model.agenerate([messages])

    actual_usage = (
        llm_result.llm_output.get("token_usage", None)
        if llm_result.llm_output
        else None
    )

    # Usage is missing when and only where streaming is enabled
    assert (actual_usage in [None, {}]) == streaming

    return llm_result.generations[0][-1].text


def create_model(
    base_url: str, model_id: str, streaming: bool
) -> BaseChatModel:
    callbacks: Callbacks = [CallbackWithNewLines()]
    return AzureChatOpenAI(
        deployment_name=model_id,
        callbacks=callbacks,
        openai_api_base=base_url,
        openai_api_version=DEFAULT_API_VERSION,
        openai_api_key="dummy_openai_api_key",
        model_kwargs={"deployment_id": model_id, "api_key": "dummy_api_key"},
        verbose=True,
        streaming=streaming,
        temperature=0.0,
        request_timeout=10,
        client=None,
        max_retries=0,
    )
