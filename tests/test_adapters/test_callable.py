from pytest_agentcheck.adapters.callable import CallableAdapter
from pytest_agentcheck.assertions import assert_max_steps, assert_output_matches
from pytest_agentcheck.recorder import Recorder


def test_callable_adapter_standalone() -> None:
    def my_agent(prompt: str) -> str:
        return f"Processed: {prompt}"

    adapter = CallableAdapter(my_agent, agent_name="test_worker")
    res, trace = adapter.run("hello world")

    assert res == "Processed: hello world"
    assert trace.agent_name == "test_worker"
    assert len(trace.steps) == 1
    assert trace.steps[0].step_type == "agent"
    assert trace.steps[0].output == "Processed: hello world"
    assert_max_steps(trace, 1)
    assert_output_matches(trace, contains="hello world")


def test_callable_adapter_with_external_recorder() -> None:
    with Recorder(agent_name="outer") as rec:
        def my_agent(val: int) -> int:
            rec.record_tool_call("multiply", {"val": val}, result=val * 2)
            return val * 2

        adapter = CallableAdapter(my_agent, recorder=rec)
        res = adapter(5)

    assert res == 10
    assert len(rec.trace.tool_calls) == 1
    assert rec.trace.tool_calls[0].name == "multiply"
    assert rec.trace.tool_calls[0].result == 10
