import pytest

from examples.mock_agent import SupportAgent, lookup_order
from pytest_agentcheck.assertions import assert_output_matches, assert_tool_called
from pytest_agentcheck.recorder import Recorder


@pytest.mark.agentcheck(runs=10, min_pass_rate=0.9, confidence=0.95)
def test_order_lookup_statistical() -> None:
    with Recorder(agent_name="support_agent") as rec:
        wrapped_lookup = rec.wrap_tool(lookup_order, tool_name="lookup_order")
        agent = SupportAgent(lookup_tool=wrapped_lookup)

        response = agent.run("Where is my order ORD-101?")
        rec.finish(output=response)

    assert_tool_called(rec.trace, "lookup_order", times=1)
    assert_output_matches(rec.trace, contains=["shipped", "ORD-101"])
    assert "FedEx" in response
