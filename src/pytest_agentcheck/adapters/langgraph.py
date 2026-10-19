from __future__ import annotations

import importlib
import json
import time
from typing import Any

from pytest_agentcheck.recorder import Recorder, get_current_recorder


class AgentCheckCallbackHandler:
    def __init__(self, recorder: Recorder | None = None) -> None:
        self.recorder = recorder
        self._tool_starts: dict[str, float] = {}
        self._tool_args: dict[str, dict[str, Any]] = {}

    def _get_recorder(self) -> Recorder | None:
        return self.recorder or get_current_recorder()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool_name = str(serialized.get("name", "tool"))
        run_id = str(kwargs.get("run_id", tool_name))
        self._tool_starts[run_id] = time.perf_counter()

        parsed_args: dict[str, Any] = {}
        try:
            parsed_args = (
                json.loads(input_str)
                if isinstance(input_str, str) and input_str.startswith("{")
                else {"input": input_str}
            )
        except Exception:
            parsed_args = {"input": input_str}

        self._tool_args[run_id] = parsed_args

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", "tool"))
        tool_name = str(kwargs.get("name", "tool"))
        t0 = self._tool_starts.pop(run_id, time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        args = self._tool_args.pop(run_id, {})

        rec = self._get_recorder()
        if rec is not None:
            rec.record_tool_call(
                name=tool_name,
                arguments=args,
                result=output,
                latency_ms=elapsed_ms,
            )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", "tool"))
        tool_name = str(kwargs.get("name", "tool"))
        t0 = self._tool_starts.pop(run_id, time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        args = self._tool_args.pop(run_id, {})

        rec = self._get_recorder()
        if rec is not None:
            rec.record_tool_call(
                name=tool_name,
                arguments=args,
                error=str(error),
                latency_ms=elapsed_ms,
            )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        rec = self._get_recorder()
        if rec is not None:
            rec.start_step(step_type="llm", input_data=prompts)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        rec = self._get_recorder()
        if rec is None:
            return

        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {})
        prompt = int(usage.get("prompt_tokens", 0))
        comp = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt + comp))
        rec.add_tokens(
            prompt_tokens=prompt,
            completion_tokens=comp,
            total_tokens=total,
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        rec = self._get_recorder()
        if rec is not None:
            rec.start_step(step_type="chain", input_data=inputs)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        rec = self._get_recorder()
        if rec is not None and rec.trace.output is None:
            rec.trace.output = outputs


def _register_langchain_callback() -> None:
    try:
        mod = importlib.import_module("langchain_core.callbacks")
        base = getattr(mod, "BaseCallbackHandler", None)
        if base is not None and hasattr(base, "register"):
            base.register(AgentCheckCallbackHandler)
    except Exception:
        pass


_register_langchain_callback()
