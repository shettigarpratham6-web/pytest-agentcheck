import json
from pathlib import Path

from typer.testing import CliRunner

from pytest_agentcheck.cli import app

runner = CliRunner()


def test_cli_report(tmp_path: Path) -> None:
    report_file = tmp_path / "report.json"
    data = {
        "summary": {
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "inconclusive": 0,
            "flaky_count": 0,
        },
        "tests": [
            {
                "nodeid": "tests/test_demo.py::test_sample",
                "successes": 10,
                "completed": 10,
                "total_runs": 10,
                "pass_rate": 1.0,
                "interval": [0.72, 1.0],
                "verdict": "PASS",
                "flaky": False,
                "early_stopped": False,
                "reason": "all_runs_completed",
            }
        ],
    }
    report_file.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(app, ["report", "-f", str(report_file)])
    assert result.exit_code == 0
    assert "test_sample" in result.stdout
    assert "PASS" in result.stdout

    result_json = runner.invoke(
        app, ["report", "-f", str(report_file), "--format", "json"]
    )
    assert result_json.exit_code == 0
    assert "total_tests" in result_json.stdout


def test_cli_diff_no_regression(tmp_path: Path) -> None:
    base_file = tmp_path / "baseline.json"
    curr_file = tmp_path / "current.json"

    item = {
        "nodeid": "test_agent",
        "successes": 95,
        "completed": 100,
        "pass_rate": 0.95,
    }
    base_file.write_text(json.dumps({"test_agent": item}), encoding="utf-8")
    curr_file.write_text(json.dumps({"test_agent": item}), encoding="utf-8")

    result = runner.invoke(app, ["diff", str(base_file), str(curr_file)])
    assert result.exit_code == 0
    assert "PASSED: No statistically significant regression" in result.stdout


def test_cli_diff_with_regression(tmp_path: Path) -> None:
    base_file = tmp_path / "baseline.json"
    curr_file = tmp_path / "current.json"

    base_item = {
        "nodeid": "test_agent",
        "successes": 98,
        "completed": 100,
        "pass_rate": 0.98,
    }
    curr_item = {
        "nodeid": "test_agent",
        "successes": 50,
        "completed": 100,
        "pass_rate": 0.50,
    }
    base_file.write_text(json.dumps({"test_agent": base_item}), encoding="utf-8")
    curr_file.write_text(json.dumps({"test_agent": curr_item}), encoding="utf-8")

    result = runner.invoke(app, ["diff", str(base_file), str(curr_file)])
    assert result.exit_code == 1
    assert "REGRESSION" in result.stdout
