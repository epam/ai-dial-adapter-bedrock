from enum import Enum
from typing import List, Tuple, Type, TypeVar

import inquirer

from llm.bedrock_models import BedrockDeployment
from llm.chat_emulation.types import ChatEmulationType

V = TypeVar("V")


def select_option(title: str, options: List[V]) -> V:
    questions = [
        inquirer.List(
            "option",
            message=title,
            choices=[(str(option), option) for option in options],
            carousel=True,
        ),
    ]
    return inquirer.prompt(questions)["option"]  # type: ignore


T = TypeVar("T", bound=Enum)


def select_enum(title: str, enum: Type[T]) -> T:
    questions = [
        inquirer.List(
            "option",
            message=title,
            choices=[(option.value, option) for option in enum],
            carousel=True,
        ),
    ]
    return inquirer.prompt(questions)["option"]  # type: ignore


def choose_deployment() -> Tuple[BedrockDeployment, ChatEmulationType]:
    deployment = select_enum("Select the deployment", BedrockDeployment)
    chat_emulation_type = select_enum(
        "Select chat emulation type", ChatEmulationType
    )

    return deployment, chat_emulation_type
