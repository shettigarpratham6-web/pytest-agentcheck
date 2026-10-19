from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass
class AgentCheckTestResult:
    nodeid: str
    successes: int
    completed: int
    total_runs: int
    pass_rate: float
    interval: tuple[float, float]
    verdict: str
    flaky: bool
    early_stopped: bool
    reason: str
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCheckTestResult:
        raw_interval = cast(list[float], data.get("interval", [0.0, 1.0]))
        interval_tuple = (
            float(raw_interval[0]),
            float(raw_interval[1]),
        )
        return cls(
            nodeid=str(data.get("nodeid", "")),
            successes=int(cast(int, data.get("successes", 0))),
            completed=int(cast(int, data.get("completed", 0))),
            total_runs=int(cast(int, data.get("total_runs", 0))),
            pass_rate=float(cast(float, data.get("pass_rate", 0.0))),
            interval=interval_tuple,
            verdict=str(data.get("verdict", "")),
            flaky=bool(data.get("flaky", False)),
            early_stopped=bool(data.get("early_stopped", False)),
            reason=str(data.get("reason", "")),
            total_tokens=int(cast(int, data.get("total_tokens", 0))),
            total_cost=float(cast(float, data.get("total_cost", 0.0))),
            total_latency_ms=float(
                cast(float, data.get("total_latency_ms", 0.0))
            ),
        )


def format_terminal_summary(results: list[AgentCheckTestResult]) -> list[str]:
    if not results:
        return []

    hdr = (
        f"{'Test':<30} {'Runs':<8} {'Pass Rate':<10} "
        f"{'Wilson 95% CI':<16} {'Verdict':<8} {'Status':<6}"
    )
    lines: list[str] = [
        "",
        "=" * 30 + " AgentCheck " + "=" * 30,
        hdr,
        "-" * 84,
    ]

    flaky_tests: list[str] = []

    for r in results:
        test_display = r.nodeid.split("::")[-1][:28]
        runs_str = f"{r.successes}/{r.completed}"
        if r.early_stopped:
            runs_str += f" (of {r.total_runs})"

        rate_str = f"{r.pass_rate * 100:.1f}%"
        ci_str = f"[{r.interval[0] * 100:.1f}%, {r.interval[1] * 100:.1f}%]"
        status_tag = "FLAKY" if r.flaky else "STABLE"
        if r.flaky:
            flaky_tests.append(f"{r.nodeid} ({rate_str})")

        row = (
            f"{test_display:<30} {runs_str:<8} {rate_str:<10} "
            f"{ci_str:<16} {r.verdict:<8} {status_tag:<6}"
        )
        lines.append(row)

    lines.append("-" * 84)

    if flaky_tests:
        lines.append("FLAKY TESTS DETECTED (Pass rate between 5% and 95%):")
        for ft in flaky_tests:
            lines.append(f"  * {ft}")
        lines.append("-" * 84)

    lines.append("=" * 72)
    return lines


def write_json_report(
    results: list[AgentCheckTestResult],
    output_dir: Path | str = ".agentcheck",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "report.json"

    data = {
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.verdict == "PASS"),
            "failed": sum(1 for r in results if r.verdict == "FAIL"),
            "inconclusive": sum(
                1 for r in results if r.verdict == "INCONCLUSIVE"
            ),
            "flaky_count": sum(1 for r in results if r.flaky),
        },
        "tests": [r.to_dict() for r in results],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return report_file


def write_markdown_report(
    results: list[AgentCheckTestResult],
    output_dir: Path | str = ".agentcheck",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "report.md"

    passed_count = sum(1 for r in results if r.verdict == "PASS")
    failed_count = sum(1 for r in results if r.verdict == "FAIL")
    flaky_count = sum(1 for r in results if r.flaky)

    lines: list[str] = [
        "# AgentCheck Evaluation Report",
        "",
        "## Summary",
        f"- **Total Tests:** {len(results)}",
        f"- **Passed:** {passed_count}",
        f"- **Failed:** {failed_count}",
        f"- **Flaky:** {flaky_count}",
        "",
        "## Statistical Results",
        "",
        "| Test | Runs | Pass Rate | Wilson 95% CI | Verdict | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        runs = (
            f"{r.successes}/{r.completed}"
            if not r.early_stopped
            else f"{r.successes}/{r.completed} (early stopped)"
        )
        rate = f"{r.pass_rate * 100:.1f}%"
        ci = f"[{r.interval[0] * 100:.1f}%, {r.interval[1] * 100:.1f}%]"
        status = "**FLAKY**" if r.flaky else "Stable"
        lines.append(
            f"| `{r.nodeid}` | {runs} | {rate} | {ci} | {r.verdict} | {status} |"
        )

    lines.append("")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report_file


def write_html_report(
    results: list[AgentCheckTestResult],
    output_dir: Path | str = ".agentcheck",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "report.html"

    rows: list[str] = []
    for r in results:
        badge_class = "pass" if r.verdict == "PASS" else "fail"
        rate = f"{r.pass_rate * 100:.1f}%"
        ci = f"[{r.interval[0] * 100:.1f}%, {r.interval[1] * 100:.1f}%]"
        flaky_text = "FLAKY" if r.flaky else "Stable"
        rows.append(
            f"<tr><td><code>{r.nodeid}</code></td>"
            f"<td>{r.successes}/{r.completed}</td>"
            f"<td>{rate}</td><td>{ci}</td>"
            f"<td><span class='badge {badge_class}'>{r.verdict}</span></td>"
            f"<td>{flaky_text}</td></tr>"
        )

    html_style = (
        "body { font-family: sans-serif; margin: 30px; background: #fafafa; }\n"
        "table { width: 100%; border-collapse: collapse; background: #fff; }\n"
        "th, td { padding: 10px; border-bottom: 1px solid #ddd; }\n"
        ".pass { background: #dcfce7; color: #166534; }\n"
        ".fail { background: #fee2e2; color: #991b1b; }"
    )

    table_header = (
        "<tr><th>Test</th><th>Runs</th><th>Pass Rate</th>"
        "<th>Wilson 95% CI</th><th>Verdict</th><th>Flakiness</th></tr>"
    )

    html_content = (
        "<!DOCTYPE html><html><head><title>AgentCheck Report</title>"
        f"<style>{html_style}</style></head><body>"
        f"<h1>AgentCheck Report</h1><table><thead>{table_header}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_file
