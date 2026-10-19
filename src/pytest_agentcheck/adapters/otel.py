from __future__ import annotations

import json
from typing import Any

from pytest_agentcheck.recorder import Recorder, get_current_recorder
from pytest_agentcheck.trace import Step, ToolCall, Trace


def ingest_spans(
    spans: list[Any],
    agent_name: str = "agent",
) -> Trace:
    trace = Trace(agent_name=agent_name)
    step_index = 0

    for span in spans:
        attrs: dict[str, Any] = {}
        if hasattr(span, "attributes") and isinstance(span.attributes, dict):
            attrs = dict(span.attributes)
        elif isinstance(span, dict):
            attrs = dict(span.get("attributes", span))

        name = str(getattr(span, "name", attrs.get("name", "span")))
        span_kind = str(attrs.get("openinference.span.kind", "")).upper()

        start_time_ns = getattr(span, "start_time", 0) or 0
        end_time_ns = getattr(span, "end_time", 0) or 0
        latency_ms = 0.0
        if end_time_ns > start_time_ns:
            latency_ms = (end_time_ns - start_time_ns) / 1_000_000.0

        if span_kind == "TOOL" or "tool.name" in attrs:
            tool_name = str(attrs.get("tool.name", name))
            raw_params = attrs.get("tool.parameters", attrs.get("input.value", {}))
            parsed_params: dict[str, Any] = {}
            if isinstance(raw_params, dict):
                parsed_params = raw_params
            elif isinstance(raw_params, str):
                try:
                    parsed_params = json.loads(raw_params)
                except Exception:
                    parsed_params = {"raw": raw_params}

            result_val = attrs.get("tool.result", attrs.get("output.value"))
            err_msg: str | None = None
            if hasattr(span, "status"):
                status_code = getattr(span.status, "status_code", None)
                if str(status_code).upper() == "ERROR":
                    err_msg = str(
                        getattr(span.status, "description", "Span error")
                    )

            tool_call = ToolCall(
                name=tool_name,
                arguments=parsed_params,
                result=result_val,
                error=err_msg,
                latency_ms=latency_ms,
            )

            step = Step(
                step_index=step_index,
                step_type="tool",
                tool_calls=[tool_call],
                latency_ms=latency_ms,
            )
            trace.steps.append(step)
            step_index += 1

        elif span_kind == "LLM":
            prompt_tokens = int(attrs.get("llm.token_count.prompt", 0))
            comp_tokens = int(attrs.get("llm.token_count.completion", 0))
            total_tokens = int(
                attrs.get(
                    "llm.token_count.total", prompt_tokens + comp_tokens
                )
            )
            cost = float(attrs.get("llm.cost", 0.0))

            step = Step(
                step_index=step_index,
                step_type="llm",
                input=attrs.get("input.value"),
                output=attrs.get("output.value"),
                tokens={
                    "prompt": prompt_tokens,
                    "completion": comp_tokens,
                    "total": total_tokens,
                },
                cost=cost,
                latency_ms=latency_ms,
            )
            trace.steps.append(step)
            step_index += 1

        elif span_kind in ("AGENT", "CHAIN"):
            if "output.value" in attrs and trace.output is None:
                trace.output = attrs["output.value"]

    return trace


class AgentCheckSpanProcessor:
    def __init__(self, recorder: Recorder | None = None) -> None:
        self.recorder = recorder

    def on_end(self, span: Any) -> None:
        rec = self.recorder or get_current_recorder()
        if rec is None:
            return

        single_trace = ingest_spans([span])
        for step in single_trace.steps:
            for call in step.tool_calls:
                rec.record_tool_call(
                    name=call.name,
                    arguments=call.arguments,
                    result=call.result,
                    error=call.error,
                    latency_ms=call.latency_ms,
                )
            if step.tokens:
                rec.add_tokens(
                    prompt_tokens=step.tokens.get("prompt", 0),
                    completion_tokens=step.tokens.get("completion", 0),
                    total_tokens=step.tokens.get("total", 0),
                    cost=step.cost,
                )
