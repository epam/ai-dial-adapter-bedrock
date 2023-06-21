from typing import List, Optional

from langchain.llms.bedrock import Bedrock
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from chat_client.init import parse_args
from llm.chat_emulation import (
    ChatEmulationType,
    history_compression,
    meta_chat_stop,
)
from utils.init import init
from utils.printing import get_input, print_ai

init()

if __name__ == "__main__":
    model_id, chat_emulation_type = parse_args()

    provider = model_id.split(".")[0]

    model_kwargs = {}
    if provider == "anthropic":
        model_kwargs["max_tokens_to_sample"] = 500

    model = Bedrock(
        model_id=model_id,
        region_name="us-east-1",
        model_kwargs=model_kwargs,
    )  # type: ignore

    stop: Optional[List[str]] = None
    if chat_emulation_type == ChatEmulationType.META_CHAT:
        stop = [meta_chat_stop]

    history: List[BaseMessage] = []

    while True:
        content = get_input("> ")
        history.append(HumanMessage(content=content))

        response = model._call(
            history_compression(chat_emulation_type, history), stop=stop
        )

        print_ai(response.strip())
        history.append(AIMessage(content=response))
