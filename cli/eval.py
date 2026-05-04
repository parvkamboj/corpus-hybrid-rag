from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="eval",
    help="Evaluate retrieval and generation quality.",
    no_args_is_help=True,
)
console = Console()

_DEFAULT_URL = "http://localhost:8000"
_DEFAULT_KEY = "dev-secret-key"
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_DATASET = "eval/golden.yaml"


@app.command()
def run(
    dataset: Path = typer.Option(
        Path(_DEFAULT_DATASET), "--dataset", "-d", help="Path to golden YAML dataset."
    ),
    strategy: str = typer.Option(
        "hybrid_rerank", "--strategy", "-s", help="Retrieval strategy to evaluate."
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Chunks to retrieve per query."),
    url: str = typer.Option(_DEFAULT_URL, "--url", help="Base URL of the running API."),
    api_key: str = typer.Option(
        _DEFAULT_KEY, "--api-key", help="X-API-Key header value."
    ),
    openai_key: str = typer.Option(
        "", "--openai-key", envvar="OPENAI_API_KEY", help="OpenAI key for RAGAS LLM."
    ),
    llm_model: str = typer.Option(
        _DEFAULT_MODEL, "--llm-model", help="LLM model for RAGAS scoring."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save JSON report to this path."
    ),
) -> None:
    """Evaluate a single strategy against a golden dataset and print RAGAS scores."""
    from eval.dataset import load_dataset
    from eval.metrics import compute_ragas
    from eval.report import print_comparison_table, save_json_report
    from eval.runner import run_eval

    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    golden = load_dataset(dataset)
    console.print(f"Loaded [bold]{golden.name}[/bold] — {len(golden)} samples")
    console.print(f"Strategy: [bold]{strategy}[/bold]  top_k={top_k}")

    with console.status("Running queries against API…"):
        samples = asyncio.run(run_eval(golden, strategy, url, api_key, top_k))

    console.print(f"Collected {len(samples)} responses. Computing RAGAS metrics…")

    with console.status("Scoring with RAGAS (this calls the LLM)…"):
        scores = compute_ragas(samples, llm_model=llm_model, openai_api_key=openai_key)

    results = {strategy: scores}
    print_comparison_table(results, dataset_name=golden.name)

    if output:
        save_json_report(results, output, dataset_name=golden.name)


@app.command()
def compare(
    dataset: Path = typer.Option(
        Path(_DEFAULT_DATASET), "--dataset", "-d", help="Path to golden YAML dataset."
    ),
    strategies: str = typer.Option(
        "dense,hybrid,hybrid_rerank",
        "--strategies",
        "-s",
        help="Comma-separated list of strategies to compare.",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Chunks to retrieve per query."),
    url: str = typer.Option(_DEFAULT_URL, "--url", help="Base URL of the running API."),
    api_key: str = typer.Option(
        _DEFAULT_KEY, "--api-key", help="X-API-Key header value."
    ),
    openai_key: str = typer.Option(
        "", "--openai-key", envvar="OPENAI_API_KEY", help="OpenAI key for RAGAS LLM."
    ),
    llm_model: str = typer.Option(
        _DEFAULT_MODEL, "--llm-model", help="LLM model for RAGAS scoring."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save JSON report to this path."
    ),
) -> None:
    """Compare multiple strategies side-by-side using RAGAS metrics."""
    from eval.dataset import load_dataset
    from eval.metrics import MetricScores, compute_ragas
    from eval.report import print_comparison_table, save_json_report
    from eval.runner import run_eval

    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
    if not strategy_list:
        console.print("[red]No strategies specified.[/red]")
        raise typer.Exit(1)

    golden = load_dataset(dataset)
    console.print(f"Loaded [bold]{golden.name}[/bold] — {len(golden)} samples")
    console.print(f"Comparing: [bold]{', '.join(strategy_list)}[/bold]  top_k={top_k}")

    all_results: dict[str, MetricScores] = {}

    for strat in strategy_list:
        console.print(f"\n[cyan]→ Running strategy:[/cyan] {strat}")
        with console.status(f"Querying API with strategy={strat}…"):
            samples = asyncio.run(run_eval(golden, strat, url, api_key, top_k))
        with console.status("Scoring with RAGAS…"):
            scores = compute_ragas(samples, llm_model=llm_model, openai_api_key=openai_key)
        all_results[strat] = scores
        console.print(
            f"  [green]✓[/green] mean={scores.mean_score():.3f}"
            f"  latency={scores.avg_latency_ms:.0f}ms"
        )

    print_comparison_table(all_results, dataset_name=golden.name)

    if output:
        save_json_report(all_results, output, dataset_name=golden.name)
