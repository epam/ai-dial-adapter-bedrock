"""
Turning a chat into a prompt for the Llama2 model.

The reference for the algo is [this code snippet](https://github.com/facebookresearch/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320-L362) in the original repository.

See also the [tokenizer](https://github.com/huggingface/transformers/blob/c99f25476312521d4425335f970b198da42f832d/src/transformers/models/llama/tokenization_llama.py#L415) in the transformers package.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.chat_emulation.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS = "<s>"
EOS = "</s>"


class Dialogue(BaseModel):
    """Valid dialog structure for LLAMA2 model:
    1. optional system message,
    2. alternating user/assistant messages,
    3. last user query"""

    system: Optional[str]
    turns: List[Tuple[str, str]]
    human: str

    def prepend_to_first_human_message(self, text: str) -> None:
        if self.turns:
            human, ai = self.turns[0]
            self.turns[0] = text + human, ai
        else:
            self.human = text + self.human


def validate_chat(messages: List[BaseMessage]) -> Dialogue:
    system: Optional[str] = None
    if messages and isinstance(messages[0], SystemMessage):
        system = messages[0].content
        if system.strip() == "":
            system = None
        messages = messages[1:]

    human = messages[::2]
    ai = messages[1::2]

    is_valid_alternation = all(
        isinstance(msg, HumanMessage) for msg in human
    ) and all(isinstance(msg, AIMessage) for msg in ai)

    if not is_valid_alternation:
        raise ValidationError(
            "The model only supports initial optional system message and"
            " follow-up alternating human/assistant messages"
        )

    turns = [
        (human.content, assistant.content)
        for human, assistant in zip(human, ai)
    ]

    if messages and isinstance(messages[-1], HumanMessage):
        last_query = messages[-1]
    else:
        raise ValidationError("The last message must be from user")

    return Dialogue(
        system=system,
        turns=turns,
        human=last_query.content,
    )


def format_sequence(text: str, bos: bool, eos: bool) -> str:
    if bos:
        text = BOS + text
    if eos:
        text = text + EOS
    return text


def create_chat_prompt(dialogue: Dialogue) -> str:
    ret: List[str] = []

    system = dialogue.system
    if system is not None:
        dialogue.prepend_to_first_human_message(B_SYS + system + E_SYS)

    ret: List[str] = [
        format_sequence(
            f"{B_INST} {human.strip()} {E_INST} {ai.strip()} ",
            bos=True,
            eos=True,
        )
        for human, ai in dialogue.turns
    ]

    ret.append(
        format_sequence(
            f"{B_INST} {dialogue.human.strip()} {E_INST}",
            bos=True,
            eos=False,
        )
    )

    return "".join(ret)


class LlamaChatEmulator(ChatEmulator):
    def display(self, messages: List[BaseMessage]) -> Tuple[str, List[str]]:
        dialogue = validate_chat(messages)
        return create_chat_prompt(dialogue), []

    def get_ai_cue(self) -> Optional[str]:
        return None


llama_emulator = LlamaChatEmulator()
