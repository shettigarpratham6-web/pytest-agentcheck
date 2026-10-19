from dataclasses import dataclass

from pytest_agentcheck.adapters.langgraph import AgentCheckCallbackHandler
from pytest_agentcheck.assertions import assert_budget, assert_tool_called
from pytest_agentcheck.recorder import Recorder


@dataclass
class MockLLMResponse:
    llm_output: dict[str, dict[str, int]]


def test_langgraph_callback_handler() -> None:
    with Recorder(agent_name="langgraph_agent") as rec:
        handler = AgentCheckCallbackHandler(recorder=rec)

        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str='{"operation": "add", "a": 5, "b": 10}',
            run_id="run_tool_1",
        )
        handler.on_tool_end(output=15, name="calculator", run_id="run_tool_1")

        llm_resp = MockLLMResponse(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                }
            }
        )
        handler.on_llm_end(llm_resp)

        handler.on_chain_end({"final_answer": "15"})

    trace = rec.trace
    assert_tool_called(trace, "calculator", times=1)
    call = trace.tool_calls[0]
    assert call.arguments == {"operation": "add", "a": 5, "b": 10}
    assert call.result == 15
    assert_budget(trace, tokens=100)
    assert trace.output == {"final_answer": "15"}


def test_langgraph_tool_error() -> None:
    with Recorder(agent_name="error_agent") as rec:
        handler = AgentCheckCallbackHandler(recorder=rec)
        handler.on_tool_start(
            serialized={"name": "fetch_db"},
            input_str="query: users",
            run_id="run_tool_err",
        )
        handler.on_tool_error(
            RuntimeError("Connection timeout"),
            name="fetch_db",
            run_id="run_tool_err",
        )

    trace = rec.trace
    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.name == "fetch_db"
    assert "Connection timeout" in str(call.error)
