from typing import List

from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def user(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def sanitize_test_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower())


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
