#!/usr/bin/env python3

import json
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.bedrock_custom import BedrockModel
from llm.bedrock_models import choose_model
from utils.init import init
from utils.printing import get_input, print_ai, print_info

if __name__ == "__main__":
    init()

    model_id, chat_emulation_type = choose_model()

    model = BedrockModel(
        model_id=model_id,
        chat_emulation_type=chat_emulation_type,
        region="us-east-1",
    )

    history: List[BaseMessage] = []

    while True:
        content = get_input("> ")
        history.append(HumanMessage(content=content))

        response = model.chat(history)

        print_info(json.dumps(response, indent=2))

        response_text = response["results"][0]["outputText"]

        print_ai(response_text.strip())
        history.append(AIMessage(content=response_text))
