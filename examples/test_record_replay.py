from pathlib import Path

from examples.mock_agent import SupportAgent, lookup_order
from pytest_agentcheck.assertions import assert_output_matches, assert_tool_called
from pytest_agentcheck.cassette import Cassette
from pytest_agentcheck.recorder import Recorder


def test_record_and_replay_offline(tmp_path: Path) -> None:
    cassette_dir = tmp_path / "cassettes"
    name = "order_status_cassette"

    with Cassette(name=name, mode="record", cassette_dir=cassette_dir) as cas:
        with Recorder(agent_name="recorder_agent") as rec:
            wrapped = rec.wrap_tool(lookup_order, tool_name="lookup_order")
            agent = SupportAgent(lookup_tool=wrapped)
            res = agent.run("Status for ORD-101")
            rec.finish(output=res)
            cas.record_interaction(
                request={"order_id": "ORD-101"},
                response={"status": "shipped", "carrier": "FedEx"},
            )

    with Cassette(name=name, mode="replay", cassette_dir=cassette_dir) as cas:
        replayed_data = cas.match_interaction({"order_id": "ORD-101"})
        assert replayed_data["status"] == "shipped"

        def replayed_tool(order_id: str) -> dict[str, str]:
            return cas.match_interaction({"order_id": order_id})

        with Recorder(agent_name="replay_agent") as rec_replay:
            wrapped_replay = rec_replay.wrap_tool(
                replayed_tool, tool_name="lookup_order"
            )
            agent_replay = SupportAgent(lookup_tool=wrapped_replay)
            res_replay = agent_replay.run("Status for ORD-101")
            rec_replay.finish(output=res_replay)

        assert_tool_called(rec_replay.trace, "lookup_order", times=1)
        assert_output_matches(rec_replay.trace, contains="shipped")
