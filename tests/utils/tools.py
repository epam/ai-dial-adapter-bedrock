from typing import List

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function

from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ai,
    ai_function,
    ai_tools,
    function_request,
    function_response,
    function_to_tool,
    sys,
    tool_request,
    tool_response,
    user,
)


class ToolCallTest:
    def __init__(self, targets: int):
        if targets == 1:
            self.cities = [("Glasgow", 15)]
        else:
            self.cities = [("Glasgow", 15), ("London", 20)]

    def messages(self, with_system: bool) -> List[ChatCompletionMessageParam]:
        query = f"Tell me what's the temperature in {' and in '.join(self.city_names)} in celsius?"
        messages: List[ChatCompletionMessageParam] = []
        if with_system:
            messages.append(sys("act as a helpful assistant"))
        messages.extend([user("2+3=?"), ai("5"), user(query)])
        return messages

    @property
    def targets(self) -> int:
        return len(self.cities)

    def get_id(self) -> str:
        return "single_cue" if self.targets == 1 else "multiple_cues"

    @property
    def city_names(self) -> List[str]:
        return [name for name, _ in self.cities]

    @property
    def city_temps(self) -> List[int]:
        return [temp for _, temp in self.cities]

    @property
    def tools(self) -> List[ChatCompletionToolParam]:
        return [function_to_tool(f) for f in self.functions]

    @property
    def functions(self) -> List[Function]:
        return [GET_WEATHER_FUNCTION]

    @property
    def function_name(self) -> str:
        return self.functions[0]["name"]

    def tool_request(self) -> ChatCompletionMessageParam:
        return ai_tools(
            [
                tool_request(
                    f"{self.function_name}_{idx+1}",
                    self.function_name,
                    self.function_args(idx),
                )
                for idx in range(self.targets)
            ]
        )

    def tool_responses(self) -> List[ChatCompletionMessageParam]:
        return [
            tool_response(
                f"{self.function_name}_{idx+1}",
                f"{self.city_temps[idx]} celsius",
            )
            for idx in range(self.targets)
        ]

    def function_request(self, idx: int) -> ChatCompletionMessageParam:
        return ai_function(
            function_request(self.function_name, self.function_args(idx))
        )

    def function_response(self, idx: int) -> ChatCompletionMessageParam:
        return function_response(
            self.function_name, f"{self.city_temps[idx]} celsius"
        )

    def function_args(self, idx: int) -> dict:
        return {
            "location": self.city_names[idx],
            "unit": "celsius",
        }

    def expected_function_args(self, idx: int) -> dict:
        return {
            "location": lambda s: self.city_names[idx].lower() in s.lower(),
            "unit": "celsius",
        }
