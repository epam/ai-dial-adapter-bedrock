import json
import logging
from typing import List

import openai
from langchain.callbacks.manager import Callbacks
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from openai.error import OpenAIError

from aidial_adapter_bedrock.utils.printing import (
    print_ai,
    print_error,
    print_info,
)
from client.conf import MAX_CHAT_TURN, MAX_INPUT_CHARS
from client.utils.args import get_client_config
from client.utils.callback import CallbackWithNewLines
from client.utils.cli import select_option
from client.utils.input import make_input

log = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2023-03-15-preview"


def get_available_models(base_url: str) -> List[str]:
    resp = openai.Model.list(
        api_type="azure",
        api_base=base_url,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )
    assert isinstance(resp, dict)
    models = [r["id"] for r in resp.get("data", [])]
    return models


def print_exception(exc: Exception) -> None:
    log.exception(exc)
    if isinstance(exc, OpenAIError):
        print_error(json.dumps(exc.json_body, indent=2))
    else:
        print_error(str(exc))


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


def main():
    client_config = get_client_config()
    base_url = f"http://{client_config.host}:{client_config.port}"

    model_id = select_option("Select the model", get_available_models(base_url))

    model = create_model(base_url, model_id, client_config.streaming)

    history: List[BaseMessage] = []

    chat_input = make_input()

    turn = 0
    while turn < MAX_CHAT_TURN:
        turn += 1

        content = chat_input()[:MAX_INPUT_CHARS]
        if content == ":clear":
            history = []
            continue
        elif content == ":quit":
            break
        elif content == ":reset":
            main()

        history.append(HumanMessage(content=content))

        try:
            llm_result = model.generate(
                [history],
                n=client_config.n,
                temperature=client_config.temperature,
            )
        except Exception as e:
            print_exception(e)
            history.pop()
            continue

        usage = (
            llm_result.llm_output.get("token_usage", {})
            if llm_result.llm_output
            else {}
        )

        ai_response: str = ""
        for generation in llm_result.generations:
            for idx, response in enumerate(generation, start=1):
                ai_response = response.text
                if not client_config.streaming:
                    print_ai(f"[{idx}] {ai_response.strip()}")

        print_info("Usage:\n" + json.dumps(usage, indent=2))

        message = AIMessage(content=ai_response)
        history.append(message)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("Exiting...")
