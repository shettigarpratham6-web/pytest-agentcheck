from __future__ import annotations

import functools
import json
import time
from typing import Any, Callable

from pytest_agentcheck.cassette import get_current_cassette
from pytest_agentcheck.recorder import Recorder, get_current_recorder


class OpenAICompatAdapter:
    def __init__(
        self,
        client: Any,
        recorder: Recorder | None = None,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self._orig_create: Callable[..., Any] | None = None
        self._patch()

    def _patch(self) -> None:
        chat = getattr(self.client, "chat", None)
        if chat is None:
            return
        completions = getattr(chat, "completions", None)
        if completions is None:
            return
        orig = getattr(completions, "create", None)
        if orig is None:
            return
        self._orig_create = orig

        adapter_self = self

        @functools.wraps(orig)
        def wrapped_create(*args: Any, **kwargs: Any) -> Any:
            cassette = get_current_cassette()
            req_data = {
                "endpoint": "chat.completions.create",
                "args": args,
                "kwargs": kwargs,
            }

            if cassette is not None and cassette.mode == "replay":
                cached = cassette.match_interaction(req_data)
                return cached.get("body", cached)

            t0 = time.perf_counter()
            resp = orig(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if cassette is not None and cassette.mode == "record":
                serialized_resp: dict[str, Any] = {}
                if hasattr(resp, "model_dump"):
                    serialized_resp = dict(resp.model_dump())
                elif isinstance(resp, dict):
                    serialized_resp = dict(resp)
                else:
                    serialized_resp = {"content": str(resp)}
                cassette.record_interaction(req_data, {"body": serialized_resp})

            adapter_self._record_llm_call(kwargs, resp, elapsed_ms)
            return resp

        setattr(completions, "create", wrapped_create)

    def unpatch(self) -> None:
        if self._orig_create is not None:
            setattr(self.client.chat.completions, "create", self._orig_create)
            self._orig_create = None

    def _record_llm_call(
        self,
        request_kwargs: dict[str, Any],
        response: Any,
        latency_ms: float,
    ) -> None:
        rec = self.recorder or get_current_recorder()
        if rec is None:
            return

        rec.start_step(step_type="llm", input_data=request_kwargs.get("messages"))

        usage = getattr(response, "usage", None)
        prompt_tokens = 0
        comp_tokens = 0
        total_tokens = 0
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
            comp_tokens = int(getattr(usage, "completion_tokens", 0))
            total_tokens = int(
                getattr(usage, "total_tokens", prompt_tokens + comp_tokens)
            )
        rec.add_tokens(
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            total_tokens=total_tokens,
        )

        choices = getattr(response, "choices", [])
        output_content: Any = None
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                output_content = getattr(msg, "content", None)
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        name = getattr(fn, "name", "tool")
                        raw_args = getattr(fn, "arguments", "{}")
                        try:
                            parsed_args = (
                                json.loads(raw_args)
                                if isinstance(raw_args, str)
                                else raw_args
                            )
                        except Exception:
                            parsed_args = {"raw": raw_args}
                        rec.record_tool_call(
                            name=str(name),
                            arguments=parsed_args,
                            latency_ms=0.0,
                        )

        rec.end_step(output=output_content, latency_ms=latency_ms)


def patch_openai_client(
    client: Any,
    recorder: Recorder | None = None,
) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(client=client, recorder=recorder)
