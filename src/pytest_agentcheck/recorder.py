from __future__ import annotations

import contextvars
import functools
import inspect
import time
from typing import Any, Callable, TypeVar, cast

from pytest_agentcheck.trace import Step, ToolCall, Trace

F = TypeVar("F", bound=Callable[..., Any])

_active_recorder: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "active_recorder", default=None
)


def get_current_recorder() -> Recorder | None:
    return _active_recorder.get()


class Recorder:
    def __init__(
        self,
        agent_name: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.trace = Trace(agent_name=agent_name, metadata=metadata or {})
        self._start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._current_step: Step | None = None
        self._token: contextvars.Token[Recorder | None] | None = None

    @classmethod
    def current(cls) -> Recorder | None:
        return get_current_recorder()

    def start(self) -> Recorder:
        self._start_time = time.perf_counter()
        self._token = _active_recorder.set(self)
        return self

    def finish(
        self, output: Any = None, error: str | None = None
    ) -> Trace:
        if self._current_step is not None:
            self.end_step()
        if self.trace.output is None and output is not None:
            self.trace.output = output
        if error is not None:
            self.trace.error = error
        if self._token is not None:
            _active_recorder.reset(self._token)
            self._token = None
        return self.trace

    def __enter__(self) -> Recorder:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        err = str(exc_val) if exc_val is not None else None
        self.finish(error=err)

    def start_step(
        self,
        step_type: str = "step",
        input_data: Any = None,
    ) -> Step:
        if self._current_step is not None:
            self.end_step()
        self._step_start_time = time.perf_counter()
        index = len(self.trace.steps)
        step = Step(
            step_index=index,
            step_type=step_type,
            input=input_data,
        )
        self._current_step = step
        return step

    def end_step(
        self,
        output: Any = None,
        tokens: dict[str, int] | None = None,
        cost: float = 0.0,
        latency_ms: float | None = None,
    ) -> Step:
        if self._current_step is None:
            self.start_step()
        step = self._current_step
        assert step is not None
        if latency_ms is None:
            latency_ms = (time.perf_counter() - self._step_start_time) * 1000.0
        step.latency_ms = latency_ms
        if output is not None:
            step.output = output
        if tokens is not None:
            step.tokens.update(tokens)
        step.cost += cost
        self.trace.steps.append(step)
        self._current_step = None
        return step

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> ToolCall:
        call = ToolCall(
            name=name,
            arguments=arguments,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )
        if self._current_step is None:
            self.start_step(step_type="tool")
        assert self._current_step is not None
        self._current_step.tool_calls.append(call)
        return call

    def add_tokens(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        cost: float = 0.0,
    ) -> None:
        if self._current_step is None:
            self.start_step(step_type="llm")
        assert self._current_step is not None
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        self._current_step.tokens["prompt"] = (
            self._current_step.tokens.get("prompt", 0) + prompt_tokens
        )
        self._current_step.tokens["completion"] = (
            self._current_step.tokens.get("completion", 0) + completion_tokens
        )
        self._current_step.tokens["total"] = (
            self._current_step.tokens.get("total", 0) + total_tokens
        )
        self._current_step.cost += cost

    def wrap_tool(
        self,
        tool_func: F,
        tool_name: str | None = None,
    ) -> F:
        name = str(tool_name or getattr(tool_func, "__name__", "tool"))

        @functools.wraps(tool_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_args: dict[str, Any] = {}
            try:
                sig = inspect.signature(tool_func)
                bound = sig.bind_partial(*args, **kwargs)
                call_args = dict(bound.arguments)
            except Exception:
                call_args = {"args": list(args), "kwargs": kwargs}

            t0 = time.perf_counter()
            err: str | None = None
            res: Any = None
            had_prior_step = self._current_step is not None
            try:
                res = tool_func(*args, **kwargs)
                return res
            except Exception as e:
                err = str(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.record_tool_call(
                    name=name,
                    arguments=call_args,
                    result=res,
                    error=err,
                    latency_ms=elapsed_ms,
                )
                if not had_prior_step:
                    self.end_step(output=res)

        return cast(F, wrapper)


def record(
    agent_name: str = "agent",
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Recorder(agent_name=agent_name, metadata=metadata) as rec:
                res = fn(*args, **kwargs)
                rec.trace.output = res
                return res

        return cast(F, wrapper)

    return decorator
