from typing import List

import inquirer


def select_option(title: str, options: List[str]) -> str:
    questions = [
        inquirer.List(
            "option",
            message=title,
            choices=[(option, option) for option in options],
            carousel=True,
        ),
    ]
    return inquirer.prompt(questions)["option"]  # type: ignore
