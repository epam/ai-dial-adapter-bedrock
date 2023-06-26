#!/usr/bin/env python3

from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.bedrock_langchain import BedrockLangChain
from llm.bedrock_models import choose_model
from utils.init import init
from utils.printing import get_input, print_ai

if __name__ == "__main__":
    init()

    model_id, chat_emulation_type = choose_model()
    model = BedrockLangChain(model_id=model_id, max_tokens=None)

    history: List[BaseMessage] = []

    while True:
        content = get_input("> ")
        history.append(HumanMessage(content=content))

        response, _ = model.chat(chat_emulation_type, history)
        print_ai(response.strip())
        history.append(AIMessage(content=response))
