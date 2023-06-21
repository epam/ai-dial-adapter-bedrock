from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from llm.callback import CallbackWithNewLines
from utils.env import get_env
from utils.init import init
from utils.printing import print_ai

if __name__ == "__main__":
    init()

    prompt_history = FileHistory(".history")

    HOST = get_env("HOST")
    PORT = int(get_env("PORT"))

    callbacks = [CallbackWithNewLines()]
    model = ChatOpenAI(
        callbacks=callbacks,
        model="amazon.titan-tg1-large",
        openai_api_base=f"http://{HOST}:{PORT}/zero_memory",
        verbose=True,
        temperature=0,
        request_timeout=6000,
    )  # type: ignore

    history: List[BaseMessage] = []

    session = PromptSession(history=prompt_history)

    while True:
        content = session.prompt("> ", style=Style.from_dict({"": "#ff0000"}))
        history.append(HumanMessage(content=content))

        response = model.generate([history]).generations[0][-1].text
        print_ai(response.strip())
        message = AIMessage(content=response)
        history.append(message)
