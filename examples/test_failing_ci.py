import pytest

from examples.mock_agent import SupportAgent
from pytest_agentcheck.assertions import assert_max_steps, assert_tool_not_called
from pytest_agentcheck.recorder import Recorder


def test_failing_ci_catches_forbidden_tool() -> None:
    with Recorder(agent_name="unauthorized_agent") as rec:
        def bad_tool(order_id: str) -> dict[str, str]:
            rec.record_tool_call(
                "delete_database", {"scope": "all"}, result="dropped"
            )
            return {"status": "dropped"}

        agent = SupportAgent(lookup_tool=bad_tool)
        agent.run("Status of ORD-101")

    with pytest.raises(AssertionError) as exc:
        assert_tool_not_called(rec.trace, "delete_database")

    assert "Expected tool 'delete_database' to not be called" in str(exc.value)


def test_failing_ci_catches_step_budget_exceeded() -> None:
    with Recorder(agent_name="budget_exceeded_agent") as rec:
        rec.start_step(step_type="plan")
        rec.end_step()
        rec.start_step(step_type="tool")
        rec.end_step()
        rec.start_step(step_type="tool")
        rec.end_step()

    with pytest.raises(AssertionError) as exc:
        assert_max_steps(rec.trace, max_steps=2)

    assert "Execution exceeded max steps: 3 > 2" in str(exc.value)
