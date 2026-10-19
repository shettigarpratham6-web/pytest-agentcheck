from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from pytest_agentcheck.recorder import Recorder, get_current_recorder
from pytest_agentcheck.trace import Trace

F = TypeVar("F", bound=Callable[..., Any])


class CallableAdapter:
    def __init__(
        self,
        agent_fn: Callable[..., Any],
        agent_name: str | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        self.agent_fn = agent_fn
        self.agent_name = str(
            agent_name or getattr(agent_fn, "__name__", "callable_agent")
        )
        self.recorder = recorder

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)[0]

    def run(self, *args: Any, **kwargs: Any) -> tuple[Any, Trace]:
        active_rec = self.recorder or get_current_recorder()
        owned = False
        if active_rec is None:
            active_rec = Recorder(agent_name=self.agent_name)
            active_rec.start()
            owned = True

        active_rec.start_step(
            step_type="agent", input_data={"args": args, "kwargs": kwargs}
        )
        t0 = time.perf_counter()
        err: str | None = None
        result: Any = None
        try:
            result = self.agent_fn(*args, **kwargs)
            return result, active_rec.trace
        except Exception as e:
            err = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            active_rec.end_step(output=result, latency_ms=elapsed_ms)
            if owned:
                active_rec.finish(output=result, error=err)
