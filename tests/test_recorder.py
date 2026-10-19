import time

from pytest_agentcheck.recorder import Recorder, get_current_recorder, record
from pytest_agentcheck.trace import Trace


def test_recorder_context_manager() -> None:
    with Recorder(agent_name="support_agent") as rec:
        assert get_current_recorder() is rec
        rec.start_step(step_type="plan", input_data="User asked for status")
        rec.record_tool_call(
            name="lookup_order",
            arguments={"order_id": "ORD-123"},
            result={"status": "shipped"},
            latency_ms=15.0,
        )
        rec.add_tokens(prompt_tokens=100, completion_tokens=25, cost=0.002)
        rec.end_step(output="Order is shipped")
        rec.finish(output="Your order ORD-123 has shipped.")

    assert get_current_recorder() is None
    trace = rec.trace
    assert trace.agent_name == "support_agent"
    assert len(trace.steps) == 1
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "lookup_order"
    assert trace.tool_calls[0].arguments == {"order_id": "ORD-123"}
    assert trace.total_tokens == 125
    assert trace.total_cost == 0.002
    assert trace.output == "Your order ORD-123 has shipped."


def test_recorder_wrap_tool() -> None:
    rec = Recorder(agent_name="calc_agent")

    def add(a: int, b: int) -> int:
        time.sleep(0.005)
        return a + b

    wrapped = rec.wrap_tool(add, tool_name="calculator_add")
    res = wrapped(2, 3)

    assert res == 5
    assert len(rec.trace.tool_calls) == 1
    call = rec.trace.tool_calls[0]
    assert call.name == "calculator_add"
    assert call.arguments == {"a": 2, "b": 3}
    assert call.result == 5
    assert call.latency_ms >= 4.0


def test_record_decorator() -> None:
    @record(agent_name="echo_agent", metadata={"env": "test"})
    def mock_agent(prompt: str) -> str:
        rec = get_current_recorder()
        assert rec is not None
        rec.record_tool_call("echo", {"text": prompt}, result=prompt)
        return f"Echo: {prompt}"

    res = mock_agent("hello")
    assert res == "Echo: hello"


def test_trace_json_roundtrip() -> None:
    with Recorder(agent_name="serial_agent") as rec:
        rec.record_tool_call("fetch", {"id": 1}, result={"data": "ok"})
        rec.add_tokens(prompt_tokens=50, completion_tokens=10)
        rec.finish(output="Done")

    json_str = rec.trace.to_json()
    loaded = Trace.from_json(json_str)

    assert loaded.schema_version == "1.0"
    assert loaded.agent_name == "serial_agent"
    assert len(loaded.steps) == 1
    assert len(loaded.tool_calls) == 1
    assert loaded.tool_calls[0].name == "fetch"
    assert loaded.tool_calls[0].result == {"data": "ok"}
    assert loaded.total_tokens == 60
    assert loaded.output == "Done"
