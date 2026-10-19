from dataclasses import dataclass, field
from typing import Any

from pytest_agentcheck.adapters.otel import AgentCheckSpanProcessor, ingest_spans
from pytest_agentcheck.assertions import assert_budget, assert_tool_called
from pytest_agentcheck.recorder import Recorder


@dataclass
class MockSpanStatus:
    status_code: str = "OK"
    description: str = ""


@dataclass
class MockSpan:
    name: str
    attributes: dict[str, Any]
    start_time: int = 1_000_000_000
    end_time: int = 1_050_000_000
    status: MockSpanStatus = field(default_factory=MockSpanStatus)


def test_ingest_spans_tool_and_llm() -> None:
    tool_span = MockSpan(
        name="web_search",
        attributes={
            "openinference.span.kind": "TOOL",
            "tool.name": "web_search",
            "tool.parameters": {"query": "weather today"},
            "tool.result": "Sunny 22C",
        },
    )

    llm_span = MockSpan(
        name="chat_model",
        attributes={
            "openinference.span.kind": "LLM",
            "input.value": "What is the weather today?",
            "output.value": "The weather is sunny at 22C.",
            "llm.token_count.prompt": 120,
            "llm.token_count.completion": 30,
            "llm.token_count.total": 150,
            "llm.cost": 0.003,
        },
    )

    agent_span = MockSpan(
        name="agent_run",
        attributes={
            "openinference.span.kind": "AGENT",
            "output.value": "The weather is sunny at 22C.",
        },
    )

    trace = ingest_spans([tool_span, llm_span, agent_span], agent_name="otel_agent")

    assert trace.agent_name == "otel_agent"
    assert len(trace.steps) == 2
    assert_tool_called(trace, "web_search", times=1)
    call = trace.tool_calls[0]
    assert call.arguments == {"query": "weather today"}
    assert call.result == "Sunny 22C"
    assert call.latency_ms == 50.0

    assert trace.total_tokens == 150
    assert_budget(trace, tokens=200, cost=0.005)
    assert trace.output == "The weather is sunny at 22C."


def test_otel_span_processor() -> None:
    with Recorder(agent_name="processor_agent") as rec:
        processor = AgentCheckSpanProcessor(recorder=rec)
        span = MockSpan(
            name="file_read",
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "read_file",
                "tool.parameters": {"path": "/tmp/test.txt"},
                "tool.result": "file contents",
            },
        )
        processor.on_end(span)

    assert_tool_called(rec.trace, "read_file", times=1)
    assert rec.trace.tool_calls[0].result == "file contents"
