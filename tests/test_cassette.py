from pathlib import Path

import pytest

from pytest_agentcheck.cassette import (
    Cassette,
    CassetteMissError,
    compute_request_hash,
    get_current_cassette,
    redact_secrets,
)
from pytest_agentcheck.faults import FaultInjector


def test_redact_secrets() -> None:
    payload = {
        "headers": {
            "Authorization": "Bearer secret_jwt_token_12345",
            "x-api-key": "sk-1234567890abcdef1234567890",
            "Content-Type": "application/json",
        },
        "body": {
            "prompt": "Here is my secret token: sk-abcdefghijklmnopqrstuvwxyz123",
            "password": "mypassword123",
            "query": "Where is ORD-101?",
        },
    }

    cleaned = redact_secrets(payload)
    assert cleaned["headers"]["Authorization"] == "[REDACTED]"
    assert cleaned["headers"]["x-api-key"] == "[REDACTED]"
    assert cleaned["headers"]["Content-Type"] == "application/json"
    assert cleaned["body"]["password"] == "[REDACTED]"
    assert "sk-abcdef" not in cleaned["body"]["prompt"]
    assert "[REDACTED_KEY]" in cleaned["body"]["prompt"]
    assert cleaned["body"]["query"] == "Where is ORD-101?"


def test_compute_request_hash_stable() -> None:
    req1 = {"tool": "order_lookup", "args": {"id": "ORD-1", "user": "alice"}}
    req2 = {"args": {"user": "alice", "id": "ORD-1"}, "tool": "order_lookup"}
    req3 = {"tool": "order_lookup", "args": {"id": "ORD-2", "user": "alice"}}

    h1 = compute_request_hash(req1)
    h2 = compute_request_hash(req2)
    h3 = compute_request_hash(req3)

    assert h1 == h2
    assert h1 != h3


def test_cassette_record_and_replay(tmp_path: Path) -> None:
    cassette_dir = tmp_path / "cassettes"
    name = "shopping_flow"

    with Cassette(name=name, mode="record", cassette_dir=cassette_dir) as cas:
        assert get_current_cassette() is cas
        cas.record_interaction(
            request={"tool": "get_order", "id": "ORD-101"},
            response={"status": "shipped"},
        )

    saved_file = cassette_dir / f"{name}.json"
    assert saved_file.exists()

    with Cassette(name=name, mode="replay", cassette_dir=cassette_dir) as cas:
        matched = cas.match_interaction(
            {"tool": "get_order", "id": "ORD-101"}
        )
        assert matched["status"] == "shipped"


def test_cassette_miss_with_diff(tmp_path: Path) -> None:
    cassette_dir = tmp_path / "cassettes"
    name = "diff_flow"

    with Cassette(name=name, mode="record", cassette_dir=cassette_dir) as cas:
        cas.record_interaction(
            request={"tool": "get_order", "id": "ORD-101"},
            response={"status": "shipped"},
        )

    with Cassette(name=name, mode="replay", cassette_dir=cassette_dir) as cas:
        with pytest.raises(CassetteMissError) as exc:
            cas.match_interaction({"tool": "get_order", "id": "ORD-999"})

        error_message = str(exc.value)
        assert "Cassette miss" in error_message
        assert "-  \"id\": \"ORD-101\"" in error_message
        assert "+  \"id\": \"ORD-999\"" in error_message


def test_replay_compatibility_with_fault_injection(tmp_path: Path) -> None:
    cassette_dir = tmp_path / "cassettes"
    name = "replay_fault"

    with Cassette(name=name, mode="record", cassette_dir=cassette_dir) as cas:
        cas.record_interaction(
            request={"tool": "fetch_user", "user_id": "U1"},
            response={"name": "Alice", "role": "admin"},
        )

    cas_replay = Cassette(name=name, mode="replay", cassette_dir=cassette_dir)

    def replayed_tool(user_id: str) -> dict[str, str]:
        return cas_replay.match_interaction(
            {"tool": "fetch_user", "user_id": user_id}
        )

    faults = FaultInjector(seed=99)
    injected_tool = faults.inject(
        replayed_tool,
        "wrong_value",
        probability=1.0,
        value={"name": "Faulty", "role": "none"},
    )

    res = injected_tool("U1")
    assert res["name"] == "Faulty"
