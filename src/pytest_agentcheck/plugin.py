from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from pytest_agentcheck.faults import FaultInjector
from pytest_agentcheck.report import (
    AgentCheckTestResult,
    format_terminal_summary,
    write_html_report,
    write_json_report,
    write_markdown_report,
)
from pytest_agentcheck.stats import (
    compare_two_proportions,
    is_flaky,
    sequential_verdict,
    wilson_score_interval,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agentcheck")
    group.addoption(
        "--agentcheck-baseline",
        action="store",
        default=None,
        choices=["save", "compare"],
        dest="agentcheck_baseline",
        help="Manage agentcheck baselines: save or compare.",
    )
    group.addoption(
        "--agentcheck-seed",
        action="store",
        default=None,
        type=int,
        dest="agentcheck_seed",
        help="Seed for reproducible agentcheck runs and fault injection.",
    )
    group.addoption(
        "--cassette-mode",
        action="store",
        default="none",
        choices=["record", "replay", "none"],
        dest="cassette_mode",
        help="Cassette recording mode: record, replay, or none.",
    )
    group.addoption(
        "--cassette-dir",
        action="store",
        default=".agentcheck/cassettes",
        dest="cassette_dir",
        help="Directory to store cassettes.",
    )
    group.addoption(
        "--agentcheck-report",
        action="store",
        default=None,
        choices=["json", "md", "html"],
        dest="agentcheck_report",
        help="Write report to .agentcheck/ in json, md, or html.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "agentcheck(runs=10, min_pass_rate=0.9, confidence=0.95): "
        "run test statistically across multiple trials",
    )
    setattr(config, "_agentcheck_results", [])


@pytest.fixture
def agentcheck_faults(request: pytest.FixtureRequest) -> FaultInjector:
    seed = request.config.getoption("agentcheck_seed")
    return FaultInjector(seed=seed)


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    marker = pyfuncitem.get_closest_marker("agentcheck")
    if marker is None:
        return None

    raw_runs = marker.kwargs.get("runs", 10)
    runs = int(raw_runs)
    raw_rate = marker.kwargs.get("min_pass_rate", 0.9)
    min_pass_rate = float(raw_rate)
    raw_conf = marker.kwargs.get("confidence", 0.95)
    confidence = float(raw_conf)

    base_seed = pyfuncitem.config.getoption("agentcheck_seed")

    successes = 0
    completed = 0
    last_error: str | None = None
    early_stopped = False
    stop_reason = ""
    verdict = "INCONCLUSIVE"

    testargs = {
        arg: pyfuncitem.funcargs[arg]
        for arg in pyfuncitem._fixtureinfo.argnames
    }

    fault_injector: Any = testargs.get("agentcheck_faults")

    for trial in range(runs):
        completed += 1
        trial_seed: int | None = None
        if base_seed is not None:
            trial_seed = int(base_seed) + trial
            random.seed(trial_seed)
            if fault_injector is not None and isinstance(
                fault_injector, FaultInjector
            ):
                fault_injector.set_seed(trial_seed)

        try:
            pyfuncitem.obj(**testargs)
            successes += 1
        except (AssertionError, Exception) as e:
            last_error = str(e)

        decision = sequential_verdict(
            successes=successes,
            completed=completed,
            total_runs=runs,
            min_pass_rate=min_pass_rate,
            confidence=confidence,
        )
        verdict = decision.verdict
        stop_reason = decision.reason

        if decision.should_stop and completed < runs:
            early_stopped = True
            break

    pass_rate = successes / completed if completed > 0 else 0.0
    interval = wilson_score_interval(successes, completed, confidence)
    flaky = is_flaky(pass_rate)

    res = AgentCheckTestResult(
        nodeid=pyfuncitem.nodeid,
        successes=successes,
        completed=completed,
        total_runs=runs,
        pass_rate=pass_rate,
        interval=interval,
        verdict=verdict,
        flaky=flaky,
        early_stopped=early_stopped,
        reason=stop_reason,
    )

    results_list: list[AgentCheckTestResult] = getattr(
        pyfuncitem.config, "_agentcheck_results", []
    )
    results_list.append(res)

    baseline_mode = pyfuncitem.config.getoption("agentcheck_baseline")
    baseline_dir = Path(".agentcheck")
    baseline_file = baseline_dir / "baselines.json"

    if baseline_mode == "save":
        baseline_dir.mkdir(parents=True, exist_ok=True)
        current_data: dict[str, Any] = {}
        if baseline_file.exists():
            with open(baseline_file, "r", encoding="utf-8") as f:
                try:
                    current_data = json.load(f)
                except Exception:
                    current_data = {}
        current_data[pyfuncitem.nodeid] = {
            "successes": successes,
            "completed": completed,
            "pass_rate": pass_rate,
            "interval": list(interval),
        }
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)

    elif baseline_mode == "compare":
        if not baseline_file.exists():
            msg = (
                f"Baseline file not found at {baseline_file}. "
                "Run with --agentcheck-baseline=save first."
            )
            raise AssertionError(msg)

        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        if pyfuncitem.nodeid not in baseline_data:
            msg = (
                f"No baseline found for test '{pyfuncitem.nodeid}'. "
                "Run with --agentcheck-baseline=save first."
            )
            raise AssertionError(msg)

        base = baseline_data[pyfuncitem.nodeid]
        comp = compare_two_proportions(
            k1=successes,
            n1=completed,
            k2=int(base["successes"]),
            n2=int(base["completed"]),
        )
        if comp.is_degradation and comp.is_significant:
            msg = (
                f"Statistical regression detected for '{pyfuncitem.nodeid}'.\n"
                f"Current pass rate: {pass_rate:.1%} ({successes}/{completed})\n"
                f"Baseline pass rate: {base['pass_rate']:.1%} "
                f"({base['successes']}/{base['completed']})\n"
                f"Difference: {comp.difference:.1%}, z-score: {comp.z_score:.2f}, "
                f"p-value: {comp.p_value:.4f}"
            )
            raise AssertionError(msg)

    else:
        if verdict == "FAIL" or (
            verdict == "INCONCLUSIVE" and pass_rate < min_pass_rate
        ):
            msg = (
                f"AgentCheck statistical test failed: pass rate {pass_rate:.1%} "
                f"({successes}/{completed}) below threshold {min_pass_rate:.1%}.\n"
                f"Wilson 95% CI: [{interval[0]:.1%}, {interval[1]:.1%}]. "
                f"Verdict: {verdict}. Reason: {stop_reason}"
            )
            if last_error:
                msg += f"\nLast trial error: {last_error}"
            raise AssertionError(msg)

    return True


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    results: list[AgentCheckTestResult] = getattr(
        config, "_agentcheck_results", []
    )
    if not results:
        return

    lines = format_terminal_summary(results)
    for line in lines:
        terminalreporter.write_line(line)

    report_format = config.getoption("agentcheck_report")
    if report_format == "json":
        path = write_json_report(results)
        terminalreporter.write_line(f"AgentCheck JSON report written to {path}")
    elif report_format == "md":
        path = write_markdown_report(results)
        terminalreporter.write_line(
            f"AgentCheck Markdown report written to {path}"
        )
    elif report_format == "html":
        path = write_html_report(results)
        terminalreporter.write_line(f"AgentCheck HTML report written to {path}")
