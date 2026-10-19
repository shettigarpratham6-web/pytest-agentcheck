from dataclasses import dataclass

from pytest_agentcheck.adapters.openai_compat import patch_openai_client
from pytest_agentcheck.assertions import assert_budget, assert_output_matches
from pytest_agentcheck.recorder import Recorder


@dataclass
class MockMessage:
    content: str


@dataclass
class MockChoice:
    message: MockMessage


@dataclass
class MockUsage:
    prompt_tokens: int = 15
    completion_tokens: int = 8
    total_tokens: int = 23


@dataclass
class MockResponse:
    choices: list[MockChoice]
    usage: MockUsage


class MockOllamaClient:
    class Chat:
        class Completions:
            def create(self, **kwargs: object) -> MockResponse:
                return MockResponse(
                    choices=[
                        MockChoice(
                            message=MockMessage(
                                content="Order ORD-101 has shipped via FedEx."
                            )
                        )
                    ],
                    usage=MockUsage(),
                )

        completions = Completions()

    chat = Chat()


def main() -> None:
    client = MockOllamaClient()

    with Recorder(agent_name="ollama_assistant") as rec:
        adapter = patch_openai_client(client, recorder=rec)

        res = client.chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": "Where is my order ORD-101?"}],
        )
        rec.finish(output=res.choices[0].message.content)

    adapter.unpatch()

    print(f"Agent response: {res.choices[0].message.content}")
    print(f"Recorded tokens: {rec.trace.total_tokens}")

    assert_budget(rec.trace, tokens=100)
    assert_output_matches(rec.trace, contains="shipped")
    print("AgentCheck assertions passed deterministically!")


if __name__ == "__main__":
    main()
