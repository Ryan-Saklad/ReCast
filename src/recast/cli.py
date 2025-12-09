"""Command-line interface for ReCast benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .benchmark import (
    run_benchmark_for_sample,
    correct_causal_graph,
    DEFAULT_DB_PATH,
)
from .dataset import (
    load_dataset as load_recast_dataset,
    get_paper_results,
    export_to_parquet,
    load_responses_from_db,
    load_evaluations_from_db,
)
from .evaluator import (
    evaluate_all_responses,
    run_aggregate_evaluation,
    run_fine_grained_evaluation,
)
from .helpers import (
    create_db,
    get_full_model_metrics,
    format_metrics_table,
)
from tabulate import tabulate

DEFAULT_CORRECTION_MODEL = "openai/gpt-4o-mini"


async def run_benchmark_async(
    model: str,
    db_path: str = DEFAULT_DB_PATH,
    provide_node_names: bool = False,
    correction_model: str | None = DEFAULT_CORRECTION_MODEL,
    sample_ids: list[int] | None = None,
    max_concurrent: int = 5,
    skip_if_exists: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 100000,
) -> None:
    """Run the benchmark for a model."""
    dataset = load_recast_dataset()
    create_db(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for item in dataset:
        cursor.execute("SELECT id FROM causal_graphs WHERE sample_id = ?", (item.id,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO causal_graphs (sample_id, domains, corrected_json, input_text) VALUES (?, ?, ?, ?)",
                (item.id, json.dumps(item.domains), json.dumps(item.edges), item.input_text),
            )
    conn.commit()
    conn.close()

    samples: list[dict[str, Any]] = []
    for item in dataset:
        if sample_ids and item.id not in sample_ids:
            continue
        samples.append({
            "id": item.id,
            "input_text": item.input_text,
            "edges": item.edges,
        })

    if not samples:
        print("No samples to process")
        return

    task_type = (
        "causal_graph_generation_with_node_names"
        if provide_node_names
        else "causal_graph_generation"
    )
    print(f"Running benchmark for {model}")
    print(f"  Task type: {task_type}")
    print(f"  Samples: {len(samples)}")
    print(f"  DB path: {db_path}")
    print(f"  Correction model: {correction_model or 'None'}")
    print()

    semaphore = asyncio.Semaphore(max_concurrent)
    successful = 0
    failed = 0

    async def run_sample(sample: dict[str, Any]) -> bool:
        async with semaphore:
            result = await run_benchmark_for_sample(
                sample_id=sample["id"],
                input_text=sample["input_text"],
                ground_truth_edges=sample["edges"],
                model=model,
                provide_node_names=provide_node_names,
                temperature=temperature,
                max_tokens=max_tokens,
                db_path=db_path,
                correction_model=correction_model,
                skip_if_exists=skip_if_exists,
            )
            return result is not None

    with tqdm(total=len(samples), desc="Processing") as pbar:
        tasks = [run_sample(s) for s in samples]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                successful += 1
            else:
                failed += 1
            pbar.update(1)

    print()
    print(f"Completed: {successful} successful, {failed} failed")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark command handler."""
    sample_ids = None
    if args.samples:
        sample_ids = [int(s) for s in args.samples.split(",")]

    asyncio.run(
        run_benchmark_async(
            model=args.model,
            db_path=args.db,
            provide_node_names=args.with_nodes,
            correction_model=args.correction_model if not args.no_correction else None,
            sample_ids=sample_ids,
            max_concurrent=args.concurrent,
            skip_if_exists=not args.no_skip,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )


def _get_results_from_source(
    source: str,
    evaluator: str,
    task_type: str,
) -> dict[str, dict] | None:
    """Load results from the specified source (huggingface, .db file, or parquet dir)."""
    source_lower = source.lower()

    # HuggingFace
    if source_lower in ("huggingface", "hf"):
        return get_paper_results(evaluator=evaluator, task_type=task_type)

    source_path = Path(source)

    # SQLite database
    if source_path.suffix == ".db" or (source_path.exists() and source_path.is_file()):
        if evaluator == "deterministic":
            return get_full_model_metrics(str(source_path), task_type)
        else:
            # For other evaluators, load from evaluations table
            from .dataset import _compute_fine_grained_metrics, _get_overall_scores
            import numpy as np

            evaluations = load_evaluations_from_db(str(source_path))
            evals = [e for e in evaluations if e.task_type == task_type]

            if evaluator == "fine_grained":
                evals = [e for e in evals if e.evaluator_type == "fine_grained"]
            elif evaluator == "llm_judge":
                evals = [e for e in evals if "llm_judge" in e.evaluator_type]
            else:
                evals = [e for e in evals if e.evaluator_type == "graph_similarity"]

            model_scores: dict[str, list[dict]] = {}
            for e in evals:
                if e.model not in model_scores:
                    model_scores[e.model] = []

                if evaluator == "llm_judge":
                    scores = e.parsed_scores
                    if scores:
                        model_scores[e.model].append(scores)
                elif evaluator == "fine_grained":
                    fg = e.parsed_fine_grained
                    if fg:
                        model_scores[e.model].append({"fine_grained": fg, "score": e.score})
                else:
                    model_scores[e.model].append({"score": e.score})

            results = {}
            for model, scores_list in model_scores.items():
                if not scores_list:
                    continue

                if evaluator == "llm_judge":
                    ca = [s.get("causal_accuracy", 0) for s in scores_list]
                    cr = [s.get("causal_recall", 0) for s in scores_list]
                    ss = [s.get("semantic_similarity", 0) for s in scores_list]
                    composite = [(a + r + s) / 15.0 for a, r, s in zip(ca, cr, ss)]
                    results[model] = {
                        "causal_accuracy": {"mean": np.mean(ca), "std": np.std(ca)},
                        "causal_recall": {"mean": np.mean(cr), "std": np.std(cr)},
                        "semantic_similarity": {"mean": np.mean(ss), "std": np.std(ss)},
                        "composite": {"mean": np.mean(composite), "std": np.std(composite)},
                        "n": len(scores_list),
                    }
                elif evaluator == "fine_grained":
                    results[model] = _compute_fine_grained_metrics(scores_list)
                else:
                    scores = [s["score"] for s in scores_list]
                    results[model] = {
                        "score": {"mean": np.mean(scores), "std": np.std(scores)},
                        "n": len(scores_list),
                    }

            return results

    # Parquet directory
    if source_path.is_dir():
        responses_path = source_path / "responses.parquet"
        evaluations_path = source_path / "evaluations.parquet"

        if not responses_path.exists() and not evaluations_path.exists():
            print(f"No parquet files found in {source_path}")
            return None

        import pandas as pd
        from .dataset import ModelResponse, Evaluation, _compute_fine_grained_metrics
        import numpy as np
        import yaml
        import re

        if evaluator == "deterministic" and responses_path.exists():
            # Compute deterministic metrics from responses
            from .helpers import calculate_full_metrics, nodes_to_node_dict

            df = pd.read_parquet(responses_path)
            dataset = load_recast_dataset()
            sample_lookup = {s.id: s for s in dataset}

            filtered = df[df["task_type"] == task_type]
            model_metrics: dict[str, dict[str, list[float]]] = {}

            for _, row in filtered.iterrows():
                sample_id = row["sample_id"]
                if sample_id not in sample_lookup:
                    continue

                sample = sample_lookup[sample_id]
                gt_edges = sample.edges
                answer = row["corrected_answer"] or row["response_answer"]
                if not answer:
                    continue

                try:
                    if "```json" in answer:
                        answer = answer.split("```json")[1].split("```")[0]
                    elif "```" in answer:
                        answer = answer.split("```")[1].split("```")[0]

                    generated = json.loads(answer)
                    if isinstance(generated, list):
                        generated = {"relationships": generated}

                    gen_edges = generated.get("relationships", [])

                    if task_type == "causal_graph_generation_with_node_names":
                        all_nodes: set[str] = set()
                        for edge in gt_edges:
                            all_nodes.add(edge["source"])
                            all_nodes.add(edge.get("sink") or edge.get("target"))

                        node_dict = nodes_to_node_dict(all_nodes)
                        id_to_name = {n["id"]: n["name"] for n in node_dict["nodes"]}

                        converted_edges = []
                        for rel in gen_edges:
                            src = rel.get("source")
                            snk = rel.get("sink") or rel.get("target")
                            if isinstance(src, int):
                                src = id_to_name.get(src, str(src))
                            if isinstance(snk, int):
                                snk = id_to_name.get(snk, str(snk))
                            converted_edges.append({"source": src, "sink": snk})
                        gen_edges = converted_edges

                    metrics = calculate_full_metrics(gt_edges, gen_edges)
                    model = row["model"]

                    if model not in model_metrics:
                        model_metrics[model] = {k: [] for k in metrics.keys()}

                    for k, v in metrics.items():
                        model_metrics[model][k].append(v)

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

            def stats(vals: list) -> dict:
                return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

            results = {}
            for model, m in model_metrics.items():
                results[model] = {
                    "node_precision": stats(m["node_precision"]),
                    "node_recall": stats(m["node_recall"]),
                    "edge_precision": stats(m["edge_precision"]),
                    "edge_recall": stats(m["edge_recall"]),
                    "f1": stats(m["f1"]),
                    "shd": stats(m["shd"]),
                    "normalized_shd": stats(m["normalized_shd"]),
                    "n": len(m["f1"]),
                }
            return results

        elif evaluations_path.exists():
            df = pd.read_parquet(evaluations_path)
            df = df[df["task_type"] == task_type]

            if evaluator == "fine_grained":
                df = df[df["evaluator_type"] == "fine_grained"]
            elif evaluator == "llm_judge":
                df = df[df["evaluator_type"].str.contains("llm_judge", na=False)]
            else:
                df = df[df["evaluator_type"] == "graph_similarity"]

            model_scores: dict[str, list[dict]] = {}
            for _, row in df.iterrows():
                model = row["model"]
                if model not in model_scores:
                    model_scores[model] = []

                if evaluator == "llm_judge":
                    try:
                        answer = row["evaluation_answer"]
                        match = re.search(r'```json\s*(\{.*?\})\s*```', answer, re.DOTALL)
                        if match:
                            data = json.loads(match.group(1))
                        else:
                            data = json.loads(answer)
                        scores = data.get("scores", data)
                        if scores:
                            model_scores[model].append(scores)
                    except:
                        pass
                elif evaluator == "fine_grained":
                    try:
                        answer = row["evaluation_answer"]
                        match = re.search(r'```yaml\s*(.*?)\s*```', answer, re.DOTALL)
                        if match:
                            fg = yaml.safe_load(match.group(1))
                        else:
                            fg = yaml.safe_load(answer)
                        if fg:
                            model_scores[model].append({"fine_grained": fg, "score": row["score"]})
                    except:
                        pass
                else:
                    model_scores[model].append({"score": row["score"]})

            results = {}
            for model, scores_list in model_scores.items():
                if not scores_list:
                    continue

                if evaluator == "llm_judge":
                    ca = [s.get("causal_accuracy", 0) for s in scores_list]
                    cr = [s.get("causal_recall", 0) for s in scores_list]
                    ss = [s.get("semantic_similarity", 0) for s in scores_list]
                    composite = [(a + r + s) / 15.0 for a, r, s in zip(ca, cr, ss)]
                    results[model] = {
                        "causal_accuracy": {"mean": np.mean(ca), "std": np.std(ca)},
                        "causal_recall": {"mean": np.mean(cr), "std": np.std(cr)},
                        "semantic_similarity": {"mean": np.mean(ss), "std": np.std(ss)},
                        "composite": {"mean": np.mean(composite), "std": np.std(composite)},
                        "n": len(scores_list),
                    }
                elif evaluator == "fine_grained":
                    results[model] = _compute_fine_grained_metrics(scores_list)
                else:
                    scores = [s["score"] for s in scores_list]
                    results[model] = {
                        "score": {"mean": np.mean(scores), "std": np.std(scores)},
                        "n": len(scores_list),
                    }

            return results

    print(f"Unknown source: {source}")
    return None


def show_results(args: argparse.Namespace) -> None:
    """Show results from any source (HuggingFace, SQLite, or parquet)."""
    if args.fine_grained:
        evaluator = "fine_grained"
    elif args.llm_judge:
        evaluator = "llm_judge"
    elif args.deterministic:
        evaluator = "deterministic"
    else:
        evaluator = "deterministic" if args.with_nodes else "fine_grained"

    task_type = (
        "causal_graph_generation_with_node_names"
        if args.with_nodes
        else "causal_graph_generation"
    )

    source = args.source
    source_display = "HuggingFace" if source.lower() in ("huggingface", "hf") else source
    print(f"Loading results from {source_display}...")

    results = _get_results_from_source(source, evaluator, task_type)

    if not results:
        print("No results found")
        return

    if evaluator == "llm_judge":
        print(f"\nReCast Results - LLM Judge")
        print("Scores are 1-5 scale; composite is normalized to 0-1\n")

        headers = ["Model", "Causal Acc", "Causal Rec", "Semantic Sim", "Composite", "N"]
        rows = []
        for model, data in sorted(
            results.items(), key=lambda x: x[1]["composite"]["mean"], reverse=True
        ):
            rows.append([
                model,
                f"{data['causal_accuracy']['mean']:.2f}±{data['causal_accuracy']['std']:.2f}",
                f"{data['causal_recall']['mean']:.2f}±{data['causal_recall']['std']:.2f}",
                f"{data['semantic_similarity']['mean']:.2f}±{data['semantic_similarity']['std']:.2f}",
                f"{data['composite']['mean']:.3f}±{data['composite']['std']:.3f}",
                data["n"],
            ])
    elif evaluator == "fine_grained":
        print(f"\nReCast Results - Fine-Grained LLM Evaluation")
        print("Metrics computed from per-node/per-edge labels\n")

        headers = ["Model", "Node Prec", "Node Rec", "Edge Prec", "Edge Rec", "F1", "N"]
        rows = []
        for model, data in sorted(
            results.items(), key=lambda x: x[1]["f1"]["mean"], reverse=True
        ):
            rows.append([
                model,
                f"{data['node_precision']['mean']:.3f}±{data['node_precision']['std']:.3f}",
                f"{data['node_recall']['mean']:.3f}±{data['node_recall']['std']:.3f}",
                f"{data['edge_precision']['mean']:.3f}±{data['edge_precision']['std']:.3f}",
                f"{data['edge_recall']['mean']:.3f}±{data['edge_recall']['std']:.3f}",
                f"{data['f1']['mean']:.3f}±{data['f1']['std']:.3f}",
                data["n"],
            ])
    elif evaluator == "deterministic":
        print(f"\nReCast Results - Deterministic Metrics")
        print("Precision/Recall/F1 computed using CDT metrics, SHD = Structural Hamming Distance\n")

        headers = ["Model", "Precision", "Recall", "F1", "SHD", "N"]
        rows = []
        for model, data in sorted(
            results.items(), key=lambda x: x[1]["f1"]["mean"], reverse=True
        ):
            rows.append([
                model,
                f"{data['edge_precision']['mean']:.3f}±{data['edge_precision']['std']:.3f}",
                f"{data['edge_recall']['mean']:.3f}±{data['edge_recall']['std']:.3f}",
                f"{data['f1']['mean']:.3f}±{data['f1']['std']:.3f}",
                f"{data['shd']['mean']:.1f}±{data['shd']['std']:.1f}",
                data.get("n") or data.get("sample_count"),
            ])
    else:
        print(f"\nReCast Results - Graph Similarity")
        print("Score is normalized SHD similarity\n")

        headers = ["Model", "Graph Similarity", "N"]
        rows = []
        for model, data in sorted(
            results.items(), key=lambda x: x[1]["score"]["mean"], reverse=True
        ):
            rows.append([
                model,
                f"{data['score']['mean']:.3f}±{data['score']['std']:.3f}",
                data["n"],
            ])

    print(tabulate(rows, headers=headers, tablefmt=args.format))


def export_results(args: argparse.Namespace) -> None:
    """Export results to JSON."""
    if args.fine_grained:
        evaluator = "fine_grained"
    elif args.llm_judge:
        evaluator = "llm_judge"
    elif args.deterministic:
        evaluator = "deterministic"
    else:
        evaluator = "deterministic" if args.with_nodes else "fine_grained"

    task_type = (
        "causal_graph_generation_with_node_names"
        if args.with_nodes
        else "causal_graph_generation"
    )

    results = _get_results_from_source(args.source, evaluator, task_type)

    if not results:
        print("No results found")
        return

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results exported to {args.output}")
    else:
        print(output)


def export_db_to_parquet(args: argparse.Namespace) -> None:
    """Export local database to parquet files."""
    output_dir = Path(args.output) if args.output else Path(".")

    print(f"Exporting {args.db} to parquet...")
    paths = export_to_parquet(
        db_path=args.db,
        output_dir=output_dir,
        include_responses=not args.evaluations_only,
        include_evaluations=not args.responses_only,
    )

    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("Done!")


async def run_evaluate_async(
    model: str,
    db_path: str = DEFAULT_DB_PATH,
    method: str = "aggregate",
    task_type: str = "causal_graph_generation",
    max_concurrent: int = 5,
    skip_if_exists: bool = True,
) -> None:
    """Run LLM-as-judge evaluations on benchmark responses."""
    method_display = "aggregate (3 metrics)" if method == "aggregate" else "fine-grained (4 evaluations)"
    print(f"Running {method_display} LLM evaluation")
    print(f"  Evaluator model: {model}")
    print(f"  Task type: {task_type}")
    print(f"  DB path: {db_path}")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Skip existing: {skip_if_exists}")
    print()

    evaluated_ids = await evaluate_all_responses(
        evaluation_method=method,
        model=model,
        task_type=task_type,
        db_path=db_path,
        max_concurrent=max_concurrent,
        skip_if_exists=skip_if_exists,
    )

    print(f"\nEvaluated {len(evaluated_ids)} responses")
    if evaluated_ids:
        print(f"Response IDs: {evaluated_ids[:10]}{'...' if len(evaluated_ids) > 10 else ''}")


def run_evaluate(args: argparse.Namespace) -> None:
    """Run LLM-as-judge evaluation command handler."""
    task_type = (
        "causal_graph_generation_with_node_names"
        if args.with_nodes
        else "causal_graph_generation"
    )

    asyncio.run(
        run_evaluate_async(
            model=args.model,
            db_path=args.db,
            method="fine_grained" if args.fine_grained else "aggregate",
            task_type=task_type,
            max_concurrent=args.concurrent,
            skip_if_exists=not args.no_skip,
        )
    )


def show_stats(args: argparse.Namespace) -> None:
    """Show dataset statistics."""
    dataset = load_recast_dataset()

    # Apply filters
    filters_applied = []
    if args.domain:
        dataset = dataset.filter_by_domain(args.domain)
        filters_applied.append(f"domain={args.domain}")
    if args.source:
        dataset = dataset.filter_by_source(args.source)
        filters_applied.append(f"source={args.source}")
    if args.after:
        dataset = dataset.filter_by_date_after(args.after)
        filters_applied.append(f"after={args.after}")
    if args.before:
        dataset = dataset.filter_by_date_before(args.before)
        filters_applied.append(f"before={args.before}")

    if len(dataset) == 0:
        print("\nNo samples match the specified filters.")
        return

    stats = dataset.statistics()

    # Header
    title = f"ReCast Dataset Statistics ({stats['total_samples']} samples)"
    if filters_applied:
        title += f" [filtered: {', '.join(filters_applied)}]"
    print(f"\n{title}\n")

    # Summary table
    dr = stats["date_range"]
    date_info = f"{dr['earliest']} to {dr['latest']}"
    if dr.get("samples_with_dates", 0) < stats["total_samples"]:
        date_info += f" ({dr['samples_with_dates']}/{stats['total_samples']} have dates)"

    summary_rows = [
        ["Date Range", date_info],
        ["Nodes", f"{stats['nodes']['min']}-{stats['nodes']['max']} (mean: {stats['nodes']['mean']})"],
        ["Edges", f"{stats['edges']['min']}-{stats['edges']['max']} (mean: {stats['edges']['mean']})"],
        ["Explicitness", f"{stats['explicitness']['min']}-{stats['explicitness']['max']} (mean: {stats['explicitness']['mean']})"],
    ]
    print(tabulate(summary_rows, tablefmt="rounded_outline"))

    if not args.brief:
        # Publication years table
        print(f"\nPublication Years:")
        year_rows = [[year, count] for year, count in stats["publication_years"].items()]
        print(tabulate(year_rows, headers=["Year", "Count"], tablefmt="rounded_outline"))

        # Domains table
        print(f"\nDomains ({len(stats['domains'])} total):")
        domain_rows = [
            [domain, count, f"{count / stats['total_samples'] * 100:.1f}%"]
            for domain, count in list(stats["domains"].items())[:15]
        ]
        print(tabulate(domain_rows, headers=["Domain", "Count", "%"], tablefmt="rounded_outline"))
        if len(stats["domains"]) > 15:
            print(f"  ... and {len(stats['domains']) - 15} more")

        # Sources table
        print(f"\nSources/Journals ({len(stats['sources'])} total):")
        source_rows = [
            [source, count, f"{count / stats['total_samples'] * 100:.1f}%"]
            for source, count in list(stats["sources"].items())[:10]
        ]
        print(tabulate(source_rows, headers=["Source", "Count", "%"], tablefmt="rounded_outline"))
        if len(stats["sources"]) > 10:
            print(f"  ... and {len(stats['sources']) - 10} more")

    if args.json:
        print(f"\nJSON Output:")
        print(json.dumps(stats, indent=2))


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="recast",
        description="ReCast: Benchmark for LLM causal reasoning on real-world academic text",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run benchmark for a model")
    run_parser.add_argument("model", help="Model identifier (e.g., deepseek/deepseek-r1)")
    run_parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    run_parser.add_argument("--with-nodes", action="store_true", help="Provide node names")
    run_parser.add_argument("--samples", help="Comma-separated sample IDs to process")
    run_parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent requests")
    run_parser.add_argument("--no-skip", action="store_true", help="Don't skip existing responses")
    run_parser.add_argument("--no-correction", action="store_true", help="Disable auto-correction")
    run_parser.add_argument(
        "--correction-model",
        default=DEFAULT_CORRECTION_MODEL,
        help="Model for correcting malformed responses",
    )
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    run_parser.add_argument("--max-tokens", type=int, default=100000, help="Max tokens to generate")
    run_parser.set_defaults(func=run_benchmark)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run LLM-as-judge evaluation on responses")
    eval_parser.add_argument(
        "model",
        nargs="?",
        default="deepseek/deepseek-r1",
        help="Evaluator model (default: deepseek/deepseek-r1)",
    )
    eval_parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    eval_parser.add_argument("--with-nodes", action="store_true", help="Evaluate name-assisted mode responses")
    eval_parser.add_argument(
        "--fine-grained",
        action="store_true",
        help="Use fine-grained evaluation (4 calls per response) instead of aggregate (1 call)",
    )
    eval_parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent evaluations")
    eval_parser.add_argument("--no-skip", action="store_true", help="Re-evaluate existing responses")
    eval_parser.set_defaults(func=run_evaluate)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--domain", help="Filter by domain (e.g., 'Macroeconomics')")
    stats_parser.add_argument("--source", help="Filter by journal source")
    stats_parser.add_argument("--after", help="Filter to samples published on/after date (YYYY-MM-DD)")
    stats_parser.add_argument("--before", help="Filter to samples published before date (YYYY-MM-DD)")
    stats_parser.add_argument("--brief", action="store_true", help="Show only summary stats")
    stats_parser.add_argument("--json", action="store_true", help="Include JSON output")
    stats_parser.set_defaults(func=show_stats)

    # Results command (unified)
    results_parser = subparsers.add_parser("results", help="Show benchmark results")
    results_parser.add_argument(
        "--source", "-s",
        default="huggingface",
        help="Data source: 'huggingface' (or 'hf'), path to .db file, or path to parquet directory",
    )
    results_parser.add_argument("--with-nodes", action="store_true", help="Show name-assisted mode results")
    eval_group = results_parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--deterministic",
        action="store_true",
        help="Show deterministic metrics (Precision/Recall/F1/SHD)",
    )
    eval_group.add_argument(
        "--fine-grained",
        action="store_true",
        help="Show fine-grained LLM evaluation (per-node/per-edge metrics)",
    )
    eval_group.add_argument(
        "--llm-judge",
        action="store_true",
        help="Show LLM judge aggregate scores (3 metrics on 1-5 scale)",
    )
    results_parser.add_argument(
        "--format",
        default="rounded_outline",
        choices=["plain", "simple", "github", "grid", "fancy_grid", "pipe", "rounded_outline"],
        help="Table format",
    )
    results_parser.set_defaults(func=show_results)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export results to JSON")
    export_parser.add_argument(
        "--source", "-s",
        default="huggingface",
        help="Data source: 'huggingface' (or 'hf'), path to .db file, or path to parquet directory",
    )
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument("--with-nodes", action="store_true", help="Filter by with-nodes mode")
    export_eval_group = export_parser.add_mutually_exclusive_group()
    export_eval_group.add_argument("--deterministic", action="store_true", help="Export deterministic metrics")
    export_eval_group.add_argument("--fine-grained", action="store_true", help="Export fine-grained metrics")
    export_eval_group.add_argument("--llm-judge", action="store_true", help="Export LLM judge metrics")
    export_parser.set_defaults(func=export_results)

    # Export DB to parquet command
    export_db_parser = subparsers.add_parser(
        "export-db",
        help="Export local database to parquet files (HuggingFace format)",
    )
    export_db_parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    export_db_parser.add_argument("--output", "-o", help="Output directory (default: current)")
    export_db_parser.add_argument(
        "--responses-only",
        action="store_true",
        help="Only export responses",
    )
    export_db_parser.add_argument(
        "--evaluations-only",
        action="store_true",
        help="Only export evaluations",
    )
    export_db_parser.set_defaults(func=export_db_to_parquet)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
