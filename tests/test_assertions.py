import pytest

from pytest_agentcheck.assertions import (
    assert_budget,
    assert_max_steps,
    assert_no_repeated_loop,
    assert_output_matches,
    assert_tool_args_match,
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_order,
)
from pytest_agentcheck.trace import Step, ToolCall, Trace


def make_trace_with_calls(calls: list[ToolCall]) -> Trace:
    return Trace(steps=[Step(step_index=0, tool_calls=calls)])


def test_assert_tool_called() -> None:
    calls = [
        ToolCall(name="search", arguments={"q": "laptop"}),
        ToolCall(name="search", arguments={"q": "phone"}),
        ToolCall(name="cart_add", arguments={"item": "laptop"}),
    ]
    trace = make_trace_with_calls(calls)

    matched = assert_tool_called(trace, "search")
    assert len(matched) == 2

    assert_tool_called(trace, "search", times=2)
    assert_tool_called(trace, "search", min_times=1, max_times=3)

    with pytest.raises(AssertionError):
        assert_tool_called(trace, "checkout")

    with pytest.raises(AssertionError):
        assert_tool_called(trace, "search", times=1)

    with pytest.raises(AssertionError):
        assert_tool_called(trace, "cart_add", min_times=2)


def test_assert_tool_not_called() -> None:
    calls = [ToolCall(name="search", arguments={})]
    trace = make_trace_with_calls(calls)

    assert_tool_not_called(trace, "delete_account")

    with pytest.raises(AssertionError):
        assert_tool_not_called(trace, "search")


def test_assert_tool_order() -> None:
    calls = [
        ToolCall(name="auth"),
        ToolCall(name="log_event"),
        ToolCall(name="fetch_profile"),
        ToolCall(name="log_event"),
        ToolCall(name="update_email"),
    ]
    trace = make_trace_with_calls(calls)

    assert_tool_order(trace, ["auth", "fetch_profile", "update_email"])

    with pytest.raises(AssertionError):
        assert_tool_order(trace, ["update_email", "fetch_profile"])

    with pytest.raises(AssertionError):
        assert_tool_order(trace, ["auth", "fetch_profile", "update_email"], exact=True)

    exact_trace = make_trace_with_calls(
        [ToolCall(name="step1"), ToolCall(name="step2")]
    )
    assert_tool_order(exact_trace, ["step1", "step2"], exact=True)


def test_assert_tool_args_match() -> None:
    calls = [
        ToolCall(
            name="lookup_order",
            arguments={"order_id": "ORD-1234", "count": 2, "active": True},
        )
    ]
    trace = make_trace_with_calls(calls)

    assert_tool_args_match(trace, "lookup_order", exact_match={"count": 2})

    with pytest.raises(AssertionError):
        assert_tool_args_match(trace, "lookup_order", exact_match={"count": 99})

    assert_tool_args_match(
        trace, "lookup_order", regex_patterns={"order_id": r"^ORD-\d+$"}
    )

    with pytest.raises(AssertionError):
        assert_tool_args_match(
            trace, "lookup_order", regex_patterns={"order_id": r"^USR-\d+$"}
        )

    schema = {
        "required": ["order_id", "count"],
        "properties": {
            "order_id": "string",
            "count": "integer",
            "active": "boolean",
        },
    }
    assert_tool_args_match(trace, "lookup_order", expected_schema=schema)

    bad_schema = {"properties": {"count": "string"}}
    with pytest.raises(AssertionError):
        assert_tool_args_match(trace, "lookup_order", expected_schema=bad_schema)


def test_assert_max_steps() -> None:
    trace = Trace(
        steps=[
            Step(step_index=0),
            Step(step_index=1),
            Step(step_index=2),
        ]
    )
    assert_max_steps(trace, 3)
    assert_max_steps(trace, 5)

    with pytest.raises(AssertionError):
        assert_max_steps(trace, 2)


def test_assert_budget() -> None:
    trace = Trace(
        steps=[
            Step(
                step_index=0,
                tokens={"prompt": 100, "completion": 50},
                cost=0.005,
                latency_ms=120.0,
            ),
            Step(
                step_index=1,
                tokens={"prompt": 50, "completion": 20},
                cost=0.002,
                latency_ms=80.0,
            ),
        ]
    )
    assert_budget(trace, tokens=250, cost=0.01, latency_ms=250.0)

    with pytest.raises(AssertionError):
        assert_budget(trace, tokens=150)

    with pytest.raises(AssertionError):
        assert_budget(trace, cost=0.005)

    with pytest.raises(AssertionError):
        assert_budget(trace, latency_ms=150.0)


def test_assert_output_matches() -> None:
    trace = Trace(
        output='{"status": "shipped", "tracking_number": "TRK-9876", "items": 2}'
    )
    assert_output_matches(trace, contains="shipped")
    assert_output_matches(trace, contains=["shipped", "TRK-9876"])
    assert_output_matches(trace, regex=r"TRK-\d{4}")

    with pytest.raises(AssertionError):
        assert_output_matches(trace, contains="delivered")

    with pytest.raises(AssertionError):
        assert_output_matches(trace, regex=r"^CANCELLED")

    schema = {"required": ["status", "tracking_number"]}
    assert_output_matches(trace, json_schema=schema)

    bad_schema = {"required": ["carrier"]}
    with pytest.raises(AssertionError):
        assert_output_matches(trace, json_schema=bad_schema)


def test_assert_no_repeated_loop() -> None:
    looping_calls = [
        ToolCall(name="search", arguments={"q": "shoes"}),
        ToolCall(name="search", arguments={"q": "shoes"}),
        ToolCall(name="search", arguments={"q": "shoes"}),
    ]
    looping_trace = make_trace_with_calls(looping_calls)
    with pytest.raises(AssertionError) as exc:
        assert_no_repeated_loop(looping_trace, max_consecutive_repeats=2)
    assert "Repeated tool loop detected" in str(exc.value)

    healthy_calls = [
        ToolCall(name="search", arguments={"q": "shoes"}),
        ToolCall(name="search", arguments={"q": "boots"}),
        ToolCall(name="search", arguments={"q": "sandals"}),
    ]
    healthy_trace = make_trace_with_calls(healthy_calls)
    assert_no_repeated_loop(healthy_trace, max_consecutive_repeats=2)
