from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from eval.metrics import MetricScores

console = Console()

_METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Ans. Relevancy"),
    ("context_precision", "Ctx. Precision"),
    ("context_recall", "Ctx. Recall"),
    ("mean_score", "Mean Score"),
    ("avg_latency_ms", "Latency (ms)"),
]


def _fmt(value: float, is_latency: bool = False) -> str:
    if is_latency:
        return f"{value:.0f}"
    color = "green" if value >= 0.7 else ("yellow" if value >= 0.4 else "red")
    return f"[{color}]{value:.3f}[/{color}]"


def print_comparison_table(
    results: dict[str, MetricScores],
    dataset_name: str = "",
) -> None:
    title = f"Evaluation Results — {dataset_name}" if dataset_name else "Evaluation Results"
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Strategy", style="bold", min_width=16)

    for _, label in _METRICS:
        table.add_column(label, justify="right", min_width=13)

    for strategy, scores in results.items():
        table.add_row(
            strategy,
            _fmt(scores.faithfulness),
            _fmt(scores.answer_relevancy),
            _fmt(scores.context_precision),
            _fmt(scores.context_recall),
            _fmt(scores.mean_score()),
            _fmt(scores.avg_latency_ms, is_latency=True),
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]Samples per strategy: "
        f"{next(iter(results.values())).sample_count if results else 0}[/dim]"
    )


def save_json_report(
    results: dict[str, MetricScores],
    path: Path,
    dataset_name: str = "",
) -> None:
    report = {
        "dataset": dataset_name,
        "strategies": {
            strategy: {
                "faithfulness": scores.faithfulness,
                "answer_relevancy": scores.answer_relevancy,
                "context_precision": scores.context_precision,
                "context_recall": scores.context_recall,
                "mean_score": scores.mean_score(),
                "avg_latency_ms": scores.avg_latency_ms,
                "sample_count": scores.sample_count,
            }
            for strategy, scores in results.items()
        },
    }
    path.write_text(json.dumps(report, indent=2))
    console.print(f"[dim]Report saved to {path}[/dim]")
