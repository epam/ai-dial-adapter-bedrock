from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from utils.files import get_project_root


def make_input(limit: int = 8000):
    session = None

    def input(prompt_text="> ", style=Style.from_dict({"": "#ff0000"})):
        nonlocal session
        if session is None:
            session = PromptSession(
                history=FileHistory(str(get_project_root() / ".history"))
            )

        response = session.prompt(prompt_text, style=style)
        return response[:limit]

    return input
