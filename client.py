from typing import List

import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from llm.callback import CallbackWithNewLines
from utils.args import get_host_port_args
from utils.cli import select_option
from utils.printing import print_ai


def get_available_models() -> List[str]:
    resp = requests.get(f"http://{host}:{port}/models").json()
    models = [r["id"] for r in resp["data"]]
    return models


if __name__ == "__main__":
    host, port = get_host_port_args()

    model = select_option("Select the model", get_available_models())

    prompt_history = FileHistory(".history")

    streaming = True
    callbacks = [CallbackWithNewLines()]
    model = ChatOpenAI(
        callbacks=callbacks,
        model=model,
        openai_api_base=f"http://{host}:{port}",
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

        response = model.generate([history]).generations[0][-1].text
        if not streaming:
            print_ai(response.strip())
        message = AIMessage(content=response)
        history.append(message)
