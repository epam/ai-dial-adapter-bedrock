import pytest

from aidial_adapter_bedrock.llm.model.claude.v1_v2.adapter import (
    get_anthropic_emulator,
)
from tests.utils.messages import ai, sys, user


@pytest.mark.parametrize("is_system_message_supported", [False, True])
def test_construction(is_system_message_supported: bool):
    messages = [
        sys(" system message1 "),
        user("  human message1  "),
        ai("     ai message1     "),
        user("  human message2  "),
    ]

    text, stop_sequences = get_anthropic_emulator(
        is_system_message_supported
    ).display(messages)

    sys_message_prefix = "Human: " if not is_system_message_supported else ""

    assert stop_sequences == ["\n\nHuman:"]
    assert text == "".join(
        [
            f"{sys_message_prefix}system message1",
            "\n\nHuman: human message1",
            "\n\nAssistant: ai message1",
            "\n\nHuman: human message2",
            "\n\nAssistant:",
        ]
    )
