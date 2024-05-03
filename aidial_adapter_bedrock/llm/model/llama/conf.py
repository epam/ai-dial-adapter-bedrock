from typing import Callable, List

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.message import BaseMessage


class LlamaConf(BaseModel):
    chat_partitioner: Callable[[List[BaseMessage]], List[int]]
    chat_emulator: ChatEmulator
