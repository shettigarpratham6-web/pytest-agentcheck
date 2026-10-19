import json
from pathlib import Path

import pytest


def test_statistical_marker_pass(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.agentcheck(runs=5, min_pass_rate=0.8)
        def test_reliable_agent():
            assert 1 + 1 == 2
        """
    )
    result = pytester.runpytest("-s")
    assert result.ret == 0
    stdout = result.stdout.str()
    assert "AgentCheck" in stdout
    assert "100.0%" in stdout
    assert "PASS" in stdout


def test_statistical_marker_fail_early_stop(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.agentcheck(runs=10, min_pass_rate=0.9)
        def test_broken_agent():
            assert False
        """
    )
    result = pytester.runpytest()
    assert result.ret != 0
    stdout = result.stdout.str()
    assert "AgentCheck statistical test failed" in stdout


def test_baseline_save_and_compare(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.agentcheck(runs=5, min_pass_rate=0.8)
        def test_baseline_agent():
            assert True
        """
    )
    res_save = pytester.runpytest("--agentcheck-baseline=save")
    assert res_save.ret == 0

    baseline_file = Path(".agentcheck/baselines.json")
    assert baseline_file.exists()
    with open(baseline_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert any("test_baseline_agent" in k for k in data.keys())

    res_compare = pytester.runpytest("--agentcheck-baseline=compare")
    assert res_compare.ret == 0


def test_report_generation(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.agentcheck(runs=3, min_pass_rate=0.6)
        def test_for_report():
            assert True
        """
    )
    result = pytester.runpytest("--agentcheck-report=json")
    assert result.ret == 0
    report_json = Path(".agentcheck/report.json")
    assert report_json.exists()

    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary"]["total_tests"] == 1
    assert data["summary"]["passed"] == 1

    result_md = pytester.runpytest("--agentcheck-report=md")
    assert result_md.ret == 0
    report_md = Path(".agentcheck/report.md")
    assert report_md.exists()
    content = report_md.read_text(encoding="utf-8")
    assert "# AgentCheck Evaluation Report" in content
