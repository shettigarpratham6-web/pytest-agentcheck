from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=str(data.get("name", "")),
            arguments=dict(data.get("arguments", {})),
            result=data.get("result"),
            error=data.get("error"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass
class Step:
    step_index: int
    step_type: str = "step"
    input: Any = None
    output: Any = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        raw_calls = data.get("tool_calls", [])
        calls = [ToolCall.from_dict(c) for c in raw_calls]
        raw_tokens = data.get("tokens", {})
        tokens = {str(k): int(v) for k, v in raw_tokens.items()}
        return cls(
            step_index=int(data.get("step_index", 0)),
            step_type=str(data.get("step_type", "step")),
            input=data.get("input"),
            output=data.get("output"),
            tool_calls=calls,
            tokens=tokens,
            latency_ms=float(data.get("latency_ms", 0.0)),
            cost=float(data.get("cost", 0.0)),
        )


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = "agent"
    steps: list[Step] = field(default_factory=list)
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    @property
    def total_tokens(self) -> int:
        total = 0
        for step in self.steps:
            if "total" in step.tokens:
                total += step.tokens["total"]
            else:
                total += sum(step.tokens.values())
        return total

    @property
    def total_cost(self) -> float:
        return sum(step.cost for step in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(step.latency_ms for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "steps": [step.to_dict() for step in self.steps],
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        raw_steps = data.get("steps", [])
        steps = [Step.from_dict(s) for s in raw_steps]
        return cls(
            trace_id=str(data.get("trace_id", str(uuid.uuid4()))),
            agent_name=str(data.get("agent_name", "agent")),
            steps=steps,
            output=data.get("output"),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
            schema_version=str(data.get("schema_version", "1.0")),
        )

    @classmethod
    def from_json(cls, text: str) -> Trace:
        return cls.from_dict(json.loads(text))
