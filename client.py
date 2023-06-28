import json
from typing import List

import requests
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from llm.callback import CallbackWithNewLines
from utils.args import get_host_port_args
from utils.cli import select_option
from utils.printing import print_ai, print_info


def get_available_models(base_url: str) -> List[str]:
    resp = requests.get(f"{base_url}/openai/models")
    if resp.status_code != 200:
        raise Exception(f"Error getting models: {resp.text}")
    resp = resp.json()
    models = [r["id"] for r in resp["data"]]
    return models


if __name__ == "__main__":
    host, port = get_host_port_args()

    base_url = f"http://{host}:{port}"
    model_id = select_option("Select the model", get_available_models(base_url))

    prompt_history = FileHistory(".history")

    streaming = select_option("Streaming?", [True, False])
    callbacks = [CallbackWithNewLines()]
    model = AzureChatOpenAI(
        deployment_name=model_id,
        callbacks=callbacks,
        openai_api_base=base_url,
        openai_api_version="2023-03-15-preview",
        verbose=True,
        streaming=streaming,
        temperature=0,
        request_timeout=6000,
    )  # type: ignore

    history: List[BaseMessage] = []

    session = PromptSession(history=prompt_history)

    while True:
        content = session.prompt("> ", style=Style.from_dict({"": "#ff0000"}))
        history.append(HumanMessage(content=content))

        llm_result = model.generate([history])

        usage = (
            llm_result.llm_output.get("token_usage", {})
            if llm_result.llm_output
            else {}
        )

        response = llm_result.generations[0][-1].text
        if not streaming:
            print_ai(response.strip())

        print_info("Usage:\n" + json.dumps(usage, indent=2))

        message = AIMessage(content=response)
        history.append(message)
