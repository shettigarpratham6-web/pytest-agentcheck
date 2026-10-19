from dataclasses import dataclass
from unittest.mock import MagicMock

from pytest_agentcheck.adapters.openai_compat import patch_openai_client
from pytest_agentcheck.assertions import assert_budget, assert_tool_called
from pytest_agentcheck.recorder import Recorder


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    function: MockFunction


@dataclass
class MockMessage:
    content: str
    tool_calls: list[MockToolCall]


@dataclass
class MockChoice:
    message: MockMessage


@dataclass
class MockUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class MockChatCompletion:
    choices: list[MockChoice]
    usage: MockUsage


class MockOpenAIClient:
    def __init__(self) -> None:
        self.chat = MagicMock()
        self.chat.completions = MagicMock()


def test_openai_compat_adapter_recording() -> None:
    mock_client = MockOpenAIClient()

    response = MockChatCompletion(
        choices=[
            MockChoice(
                message=MockMessage(
                    content="The weather in Paris is sunny.",
                    tool_calls=[
                        MockToolCall(
                            function=MockFunction(
                                name="get_weather",
                                arguments='{"city": "Paris"}',
                            )
                        )
                    ],
                )
            )
        ],
        usage=MockUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
    )
    mock_client.chat.completions.create = MagicMock(return_value=response)

    with Recorder(agent_name="openai_agent") as rec:
        adapter = patch_openai_client(mock_client, recorder=rec)
        res = mock_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
        )

    adapter.unpatch()
    assert res == response
    assert len(rec.trace.steps) == 1
    assert rec.trace.steps[0].step_type == "llm"
    assert rec.trace.total_tokens == 55
    assert_budget(rec.trace, tokens=100)
    assert_tool_called(rec.trace, "get_weather", times=1)
    call = rec.trace.tool_calls[0]
    assert call.arguments == {"city": "Paris"}
