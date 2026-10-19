from __future__ import annotations

import json
import re
from typing import Any

from pytest_agentcheck.trace import ToolCall, Trace


def assert_tool_called(
    trace: Trace,
    tool_name: str,
    times: int | None = None,
    min_times: int | None = None,
    max_times: int | None = None,
) -> list[ToolCall]:
    matching = [call for call in trace.tool_calls if call.name == tool_name]
    count = len(matching)

    if count == 0:
        actual_names = [call.name for call in trace.tool_calls]
        msg = (
            f"Expected tool '{tool_name}' to be called, but it was not. "
            f"Called tools: {actual_names}"
        )
        raise AssertionError(msg)

    if times is not None and count != times:
        msg = (
            f"Expected tool '{tool_name}' to be called exactly {times} times, "
            f"but got {count}"
        )
        raise AssertionError(msg)

    if min_times is not None and count < min_times:
        msg = (
            f"Expected tool '{tool_name}' to be called at least {min_times} "
            f"times, but got {count}"
        )
        raise AssertionError(msg)

    if max_times is not None and count > max_times:
        msg = (
            f"Expected tool '{tool_name}' to be called at most {max_times} "
            f"times, but got {count}"
        )
        raise AssertionError(msg)

    return matching


def assert_tool_not_called(trace: Trace, tool_name: str) -> None:
    matching = [call for call in trace.tool_calls if call.name == tool_name]
    if matching:
        msg = (
            f"Expected tool '{tool_name}' to not be called, "
            f"but it was called {len(matching)} time(s)"
        )
        raise AssertionError(msg)


def assert_tool_order(
    trace: Trace,
    expected_order: list[str],
    exact: bool = False,
) -> None:
    actual = [call.name for call in trace.tool_calls]

    if exact:
        if actual != expected_order:
            msg = (
                "Tool calls did not match exact expected order.\n"
                f"Expected: {expected_order}\n"
                f"Actual:   {actual}"
            )
            raise AssertionError(msg)
        return

    pos = 0
    for name in actual:
        if pos < len(expected_order) and name == expected_order[pos]:
            pos += 1

    if pos != len(expected_order):
        msg = (
            "Tool calls did not match expected subsequence.\n"
            f"Expected: {expected_order}\n"
            f"Actual:   {actual}"
        )
        raise AssertionError(msg)


def assert_tool_args_match(
    trace: Trace,
    tool_name: str,
    expected_schema: dict[str, Any] | None = None,
    regex_patterns: dict[str, str] | None = None,
    exact_match: dict[str, Any] | None = None,
) -> None:
    matching = assert_tool_called(trace, tool_name)

    for call in matching:
        args = call.arguments

        if exact_match is not None:
            for k, v in exact_match.items():
                if k not in args or args[k] != v:
                    msg = (
                        f"Tool '{tool_name}' arguments did not match exact values.\n"
                        f"Expected {k}={v}, got {args.get(k)}"
                    )
                    raise AssertionError(msg)

        if regex_patterns is not None:
            for k, pattern in regex_patterns.items():
                val = str(args.get(k, ""))
                if not re.search(pattern, val):
                    msg = (
                        f"Tool '{tool_name}' arg '{k}' ('{val}') "
                        f"did not match regex pattern '{pattern}'"
                    )
                    raise AssertionError(msg)

        if expected_schema is not None:
            type_map: dict[str, tuple[type, ...] | type] = {
                "string": str,
                "str": str,
                "integer": int,
                "int": int,
                "number": (int, float),
                "float": float,
                "boolean": bool,
                "bool": bool,
                "array": list,
                "list": list,
                "object": dict,
                "dict": dict,
            }
            props = expected_schema.get("properties", expected_schema)
            for k, expected_type in props.items():
                if k not in args:
                    if (
                        "required" in expected_schema
                        and k in expected_schema["required"]
                    ):
                        raise AssertionError(
                            f"Missing required argument '{k}' for tool '{tool_name}'"
                        )
                    continue

                target_type: Any = expected_type
                if isinstance(expected_type, dict) and "type" in expected_type:
                    target_type = expected_type["type"]

                if isinstance(target_type, str) and target_type in type_map:
                    py_type = type_map[target_type]
                    if not isinstance(args[k], py_type):
                        actual_type = type(args[k]).__name__
                        msg = (
                            f"Argument '{k}' of tool '{tool_name}' had invalid type. "
                            f"Expected {target_type}, got {actual_type}"
                        )
                        raise AssertionError(msg)


def assert_max_steps(trace: Trace, max_steps: int) -> None:
    actual = len(trace.steps)
    if actual > max_steps:
        raise AssertionError(
            f"Execution exceeded max steps: {actual} > {max_steps}"
        )


def assert_budget(
    trace: Trace,
    tokens: int | None = None,
    cost: float | None = None,
    latency_ms: float | None = None,
) -> None:
    if tokens is not None and trace.total_tokens > tokens:
        raise AssertionError(
            f"Budget exceeded for tokens: {trace.total_tokens} > {tokens}"
        )
    if cost is not None and trace.total_cost > cost:
        raise AssertionError(
            f"Budget exceeded for cost: ${trace.total_cost:.4f} > ${cost:.4f}"
        )
    if latency_ms is not None and trace.total_latency_ms > latency_ms:
        msg = (
            f"Budget exceeded for latency: "
            f"{trace.total_latency_ms:.1f}ms > {latency_ms:.1f}ms"
        )
        raise AssertionError(msg)


def assert_output_matches(
    trace: Trace,
    contains: str | list[str] | None = None,
    regex: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> None:
    raw_output = trace.output
    text = str(raw_output if raw_output is not None else "")

    if contains is not None:
        needles = [contains] if isinstance(contains, str) else contains
        for needle in needles:
            if needle.lower() not in text.lower():
                msg = (
                    f"Output did not contain expected text '{needle}'. "
                    f"Output was:\n{text}"
                )
                raise AssertionError(msg)

    if regex is not None:
        if not re.search(regex, text):
            raise AssertionError(
                f"Output did not match regex '{regex}'. Output was:\n{text}"
            )

    if json_schema is not None:
        parsed: Any = None
        if isinstance(raw_output, dict):
            parsed = raw_output
        else:
            try:
                parsed = json.loads(text)
            except Exception as e:
                raise AssertionError(
                    f"Output is not valid JSON: {e}. Output was:\n{text}"
                ) from e

        req = json_schema.get("required", [])
        for k in req:
            if k not in parsed:
                raise AssertionError(
                    f"JSON output missing required field '{k}'"
                )


def assert_no_repeated_loop(
    trace: Trace,
    max_consecutive_repeats: int = 2,
) -> None:
    calls = trace.tool_calls
    if len(calls) < max_consecutive_repeats:
        return

    consecutive = 1
    last_key: str | None = None

    for call in calls:
        current_key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
        if current_key == last_key:
            consecutive += 1
            if consecutive >= max_consecutive_repeats:
                msg = (
                    f"Repeated tool loop detected: '{call.name}' called "
                    f"{consecutive} times consecutively with identical arguments: "
                    f"{call.arguments}"
                )
                raise AssertionError(msg)
        else:
            consecutive = 1
            last_key = current_key
