from __future__ import annotations

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
from pytest_agentcheck.cassette import (
    Cassette,
    CassetteMissError,
    compute_request_hash,
    get_current_cassette,
    redact_secrets,
)
from pytest_agentcheck.faults import (
    AgentCheckFaultError,
    FaultInjectedException,
    FaultInjector,
    FaultRateLimitError,
    FaultTimeoutError,
)
from pytest_agentcheck.recorder import Recorder, get_current_recorder, record
from pytest_agentcheck.stats import (
    compare_two_proportions,
    flakiness_score,
    is_flaky,
    sequential_verdict,
    wilson_score_interval,
)
from pytest_agentcheck.trace import Step, ToolCall, Trace

__version__ = "0.1.1"

__all__ = [
    "AgentCheckFaultError",
    "Cassette",
    "CassetteMissError",
    "FaultInjectedException",
    "FaultInjector",
    "FaultRateLimitError",
    "FaultTimeoutError",
    "Recorder",
    "Step",
    "ToolCall",
    "Trace",
    "assert_budget",
    "assert_max_steps",
    "assert_no_repeated_loop",
    "assert_output_matches",
    "assert_tool_args_match",
    "assert_tool_called",
    "assert_tool_not_called",
    "assert_tool_order",
    "compare_two_proportions",
    "compute_request_hash",
    "flakiness_score",
    "get_current_cassette",
    "get_current_recorder",
    "is_flaky",
    "record",
    "redact_secrets",
    "sequential_verdict",
    "wilson_score_interval",
]
