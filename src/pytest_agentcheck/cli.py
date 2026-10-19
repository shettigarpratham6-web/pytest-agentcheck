from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from pytest_agentcheck.report import AgentCheckTestResult
from pytest_agentcheck.stats import compare_two_proportions

app = typer.Typer(
    name="agentcheck",
    help="AgentCheck CLI for reports and diffs",
    no_args_is_help=True,
)
console = Console()


@app.command("report")
def report_command(
    file: str = typer.Option(
        ".agentcheck/report.json",
        "--file",
        "-f",
        help="Path to report JSON file",
    ),
    format: str = typer.Option(
        "terminal",
        "--format",
        help="Output format: terminal, json, or markdown",
    ),
) -> None:
    path = Path(file)
    if not path.exists():
        console.print(f"[red]Report file not found: {path}[/red]")
        raise typer.Exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_tests = data.get("tests", [])
    results = [AgentCheckTestResult.from_dict(t) for t in raw_tests]

    if format == "json":
        console.print_json(json.dumps(data))
        return

    if format == "markdown":
        console.print("# AgentCheck Report")
        for r in results:
            console.print(
                f"- **{r.nodeid}**: {r.pass_rate * 100:.1f}% ({r.verdict})"
            )
        return

    table = Table(title="AgentCheck Statistical Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Wilson 95% CI", justify="center")
    table.add_column("Verdict", justify="center")
    table.add_column("Status", justify="center")

    for r in results:
        verdict_style = "green" if r.verdict == "PASS" else "red"
        status_style = "yellow" if r.flaky else "dim"
        table.add_row(
            r.nodeid.split("::")[-1],
            f"{r.successes}/{r.completed}",
            f"{r.pass_rate * 100:.1f}%",
            f"[{r.interval[0] * 100:.1f}%, {r.interval[1] * 100:.1f}%]",
            f"[{verdict_style}]{r.verdict}[/{verdict_style}]",
            f"[{status_style}]{'FLAKY' if r.flaky else 'STABLE'}[/{status_style}]",
        )

    console.print(table)


@app.command("diff")
def diff_command(
    baseline: str = typer.Argument(..., help="Path to baseline file (.json)"),
    current: str = typer.Argument(..., help="Path to current report (.json)"),
    alpha: float = typer.Option(
        0.05, "--alpha", "-a", help="Significance level"
    ),
) -> None:
    base_path = Path(baseline)
    curr_path = Path(current)

    if not base_path.exists():
        console.print(f"[red]Baseline file not found: {base_path}[/red]")
        raise typer.Exit(1)
    if not curr_path.exists():
        console.print(f"[red]Current file not found: {curr_path}[/red]")
        raise typer.Exit(1)

    with open(base_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(curr_path, "r", encoding="utf-8") as f:
        curr_data = json.load(f)

    base_map: dict[str, dict[str, Any]] = {}
    if isinstance(base_data, dict) and "tests" in base_data:
        for item in base_data["tests"]:
            base_map[item["nodeid"]] = item
    elif isinstance(base_data, dict):
        base_map = base_data

    curr_map: dict[str, dict[str, Any]] = {}
    if isinstance(curr_data, dict) and "tests" in curr_data:
        for item in curr_data["tests"]:
            curr_map[item["nodeid"]] = item
    elif isinstance(curr_data, dict):
        curr_map = curr_data

    table = Table(title="AgentCheck Baseline Regression Comparison")
    table.add_column("Test", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Status", justify="center")

    has_regression = False

    for nodeid, curr_item in curr_map.items():
        if nodeid not in base_map:
            continue

        base_item = base_map[nodeid]
        k1 = int(curr_item["successes"])
        n1 = int(curr_item["completed"])
        k2 = int(base_item["successes"])
        n2 = int(base_item["completed"])

        res = compare_two_proportions(k1, n1, k2, n2, alpha=alpha)

        base_pct = f"{res.p2 * 100:.1f}%"
        curr_pct = f"{res.p1 * 100:.1f}%"
        diff_pct = f"{res.difference * 100:+.1f}%"
        pval_str = f"{res.p_value:.4f}"

        if res.is_degradation and res.is_significant:
            has_regression = True
            status_str = "[red bold]REGRESSION[/red bold]"
        elif res.is_degradation:
            status_str = "[yellow]DROP (NOISE)[/yellow]"
        else:
            status_str = "[green]STABLE[/green]"

        table.add_row(
            nodeid.split("::")[-1],
            base_pct,
            curr_pct,
            diff_pct,
            pval_str,
            status_str,
        )

    console.print(table)
    if has_regression:
        console.print(
            "[red bold]FAILED: Statistically significant "
            "regression detected![/red bold]"
        )
        raise typer.Exit(1)
    console.print("[green]PASSED: No statistically significant regression.[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
