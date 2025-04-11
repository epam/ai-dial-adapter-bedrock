import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, assert_never

import httpx
import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.region_deployment import RegionDeployment
from tests.integration_tests.test_chat_completion import chat_deployments
from tests.utils.json import match_objects
from tests.utils.openai import (
    ai,
    sanitize_test_name,
    sys,
    tokenize,
    truncate_prompt,
    user,
)


class ExpectedException(BaseModel):
    type: type[Exception]
    message: str
    status_code: int | None = None


def expected_success(*args, **kwargs):
    return True


Tokenize = Callable[[List[ChatCompletionMessageParam]], Awaitable[int]]


@dataclass
class DynamicMaxPromptTokens:
    messages: List[ChatCompletionMessageParam]
    post_process: Callable[[int], int] | None = None

    async def __call__(self, tokenize: Tokenize) -> int:
        tokens = await tokenize(self.messages)
        if self.post_process:
            tokens = self.post_process(tokens)
        return tokens


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: RegionDeployment
    messages: List[ChatCompletionMessageParam]
    expected: dict | Callable[[dict], bool] | ExpectedException
    max_prompt_tokens: DynamicMaxPromptTokens | int | None

    def get_id(self):
        if isinstance(self.max_prompt_tokens, int):
            max_prompt_tokens_str = str(self.max_prompt_tokens)
        elif isinstance(self.max_prompt_tokens, list):
            max_prompt_tokens_str = "dynamic"
        else:
            max_prompt_tokens_str = "none"

        return sanitize_test_name(
            f"{self.deployment.value}/maxpt:{max_prompt_tokens_str}/{self.name}"
        )


def get_test_cases(deployment: RegionDeployment) -> List[TestCase]:
    test_cases: List[TestCase] = []

    def test_case(
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: (
            dict | Callable[[dict], bool] | ExpectedException
        ) = expected_success,
        max_prompt_tokens: DynamicMaxPromptTokens | int | None = None,
    ) -> None:
        test_cases.append(
            TestCase(name, deployment, messages, expected, max_prompt_tokens)
        )

    test_case(
        name="invalid request",
        messages=[
            {"role": "foo", "content": "bar"},  # type: ignore
        ],
        expected=ExpectedException(
            type=httpx.HTTPStatusError,
            message=json.dumps(
                {
                    "error": {
                        "message": "Your request contained invalid structure on path inputs.0.messages.0.role. value is not a valid enumeration member; permitted: 'system', 'developer', 'user', 'assistant', 'function', 'tool'",
                        "type": "invalid_request_error",
                        "code": "400",
                    }
                },
                separators=(",", ":"),
            ),
            status_code=400,
        ),
    )

    test_case(
        name="no max_prompt_tokens",
        messages=[
            user("ping"),
            ai("pong"),
            user("test"),
        ],
        expected={
            "outputs": [
                {"status": "error", "error": "max_prompt_tokens is required"}
            ]
        },
    )

    test_case(
        name="keep all",
        messages=[
            user("ping"),
            ai("pong"),
            user("test"),
        ],
        max_prompt_tokens=DynamicMaxPromptTokens(
            [
                user("ping"),
                ai("pong"),
                user("test"),
            ],
        ),
        expected={"outputs": [{"status": "success", "discarded_messages": []}]},
    )

    test_case(
        name="max_prompt_tokens is too small",
        messages=[
            user("ping"),
            ai("pong"),
            user("hello world"),
        ],
        max_prompt_tokens=DynamicMaxPromptTokens(
            [user("hello world")], lambda x: x - 1
        ),
        expected={
            "outputs": [
                {
                    "status": "error",
                    "error": re.compile(
                        r"The requested maximum prompt tokens is \d+. However, the system messages and the last user message resulted in \d+ tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
                    ),
                }
            ]
        },
    )

    test_case(
        name="keep last user message",
        messages=[
            user("ping"),
            ai("pong"),
            user("hello world"),
        ],
        max_prompt_tokens=DynamicMaxPromptTokens([user("hello world")]),
        expected={
            "outputs": [{"status": "success", "discarded_messages": [0, 1]}]
        },
    )

    test_case(
        name="empty sys + keep last user message",
        messages=[
            sys(""),
            sys(""),
            user("ping"),
            ai("pong"),
            user("hello world"),
        ],
        max_prompt_tokens=DynamicMaxPromptTokens([user("hello world")]),
        expected={
            "outputs": [
                {"status": "success", "discarded_messages": [0, 1, 2, 3]}
            ]
        },
    )

    return test_cases


@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment, _region in chat_deployments.items()
        for test in get_test_cases(deployment)
    ],
    ids=lambda test: test.get_id(),
)
async def test_truncate_prompt(
    test_http_client: httpx.AsyncClient, test: TestCase
):
    async def run_truncate_prompt() -> dict:
        if test.max_prompt_tokens is None:
            max_prompt_tokens = None
        elif isinstance(test.max_prompt_tokens, int):
            max_prompt_tokens = test.max_prompt_tokens
        elif callable(test.max_prompt_tokens):

            async def tokenize_messages(
                messages: List[ChatCompletionMessageParam],
            ) -> int:
                resp = await tokenize(
                    test_http_client, test.deployment.value, messages
                )
                return resp["outputs"][0]["token_count"]

            max_prompt_tokens = await test.max_prompt_tokens(tokenize_messages)
        else:
            assert_never(test.max_prompt_tokens)

        return await truncate_prompt(
            test_http_client,
            test.deployment.value,
            test.messages,
            max_prompt_tokens,
        )

    if isinstance(test.expected, ExpectedException):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await run_truncate_prompt()

        actual_exc = exc_info.value

        assert isinstance(
            actual_exc, test.expected.type
        ), f"Actual exception type ({type(actual_exc)}) doesn't match the expected one ({test.expected.type})"
        assert test.expected.status_code == actual_exc.response.status_code
        assert re.search(test.expected.message, actual_exc.response.text)
    else:
        actual_output = await run_truncate_prompt()

        if isinstance(test.expected, dict):
            match_objects(test.expected, actual_output)
        else:
            assert test.expected(
                actual_output
            ), f"Failed output test, actual output: {actual_output}"
