from __future__ import annotations

import contextvars
import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SECRET_KEYS = {
    "authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "secret",
    "token",
    "password",
    "proxy-authorization",
}

BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]+", re.IGNORECASE)
API_KEY_PATTERN = re.compile(
    r"\b(?:sk|key|ghp|token|secret)-[A-Za-z0-9_\-]{16,}\b", re.IGNORECASE
)


class CassetteMissError(AssertionError):
    pass


def redact_secrets(data: Any) -> Any:
    if isinstance(data, dict):
        redacted: dict[str, Any] = {}
        for k, v in data.items():
            if str(k).lower() in SECRET_KEYS:
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = redact_secrets(v)
        return redacted

    if isinstance(data, list):
        return [redact_secrets(item) for item in data]

    if isinstance(data, str):
        val = BEARER_PATTERN.sub("Bearer [REDACTED]", data)
        val = API_KEY_PATTERN.sub("[REDACTED_KEY]", val)
        return val

    return data


def compute_request_hash(request_data: dict[str, Any]) -> str:
    cleaned = redact_secrets(request_data)
    canonical = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class Interaction:
    request: dict[str, Any]
    response: dict[str, Any]
    request_hash: str = ""

    def __post_init__(self) -> None:
        if not self.request_hash:
            self.request_hash = compute_request_hash(self.request)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "request": redact_secrets(self.request),
            "response": redact_secrets(self.response),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interaction:
        return cls(
            request=dict(data.get("request", {})),
            response=dict(data.get("response", {})),
            request_hash=str(data.get("request_hash", "")),
        )


_active_cassette: contextvars.ContextVar[Cassette | None] = contextvars.ContextVar(
    "active_cassette", default=None
)


def get_current_cassette() -> Cassette | None:
    return _active_cassette.get()


class Cassette:
    def __init__(
        self,
        name: str,
        mode: str = "none",
        cassette_dir: str | Path = ".agentcheck/cassettes",
    ) -> None:
        self.name = name
        self.mode = mode
        self.cassette_dir = Path(cassette_dir)
        self.interactions: list[Interaction] = []
        self._played_indices: set[int] = set()
        self._token: contextvars.Token[Cassette | None] | None = None
        self._filepath = self.cassette_dir / f"{name}.json"

        if self.mode == "replay":
            self.load()

    def __enter__(self) -> Cassette:
        self._token = _active_cassette.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._token is not None:
            _active_cassette.reset(self._token)
            self._token = None
        if self.mode == "record" and exc_val is None:
            self.save()

    def record_interaction(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> Interaction:
        interaction = Interaction(
            request=redact_secrets(request),
            response=redact_secrets(response),
        )
        self.interactions.append(interaction)
        return interaction

    def match_interaction(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        req_hash = compute_request_hash(request)

        for i, inter in enumerate(self.interactions):
            if i not in self._played_indices and inter.request_hash == req_hash:
                self._played_indices.add(i)
                return dict(inter.response)

        for i, inter in enumerate(self.interactions):
            if inter.request_hash == req_hash:
                return dict(inter.response)

        diff = self._build_diff(request)
        msg = (
            f"Cassette miss for '{self.name}'. "
            f"No matching interaction found for request hash '{req_hash}'.\n"
            f"Diff against closest interaction:\n{diff}"
        )
        raise CassetteMissError(msg)

    def _build_diff(self, incoming_request: dict[str, Any]) -> str:
        if not self.interactions:
            return "(Cassette is empty)"

        incoming_json = json.dumps(
            redact_secrets(incoming_request), indent=2, sort_keys=True
        )
        closest_json = json.dumps(
            self.interactions[0].request, indent=2, sort_keys=True
        )

        diff_lines = list(
            difflib.unified_diff(
                closest_json.splitlines(),
                incoming_json.splitlines(),
                fromfile="recorded",
                tofile="incoming",
                lineterm="",
            )
        )
        return "\n".join(diff_lines) if diff_lines else "(Identical structure)"

    def save(self) -> None:
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "version": "1.0",
            "interactions": [inter.to_dict() for inter in self.interactions],
        }
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if not self._filepath.exists():
            msg = (
                f"Cassette file not found: {self._filepath}. "
                "Run in --cassette-mode=record first."
            )
            raise FileNotFoundError(msg)

        with open(self._filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_interactions = data.get("interactions", [])
        self.interactions = [
            Interaction.from_dict(item) for item in raw_interactions
        ]
        self._played_indices.clear()
