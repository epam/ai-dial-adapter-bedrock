from typing import List

from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage


def emulate(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise ValidationError("List of messages must not be empty")
    return prompt[-1].content
