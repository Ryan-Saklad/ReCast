"""Dataset loading utilities for ReCast benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import json
import re
import sqlite3

import yaml


@dataclass
class ReCastSample:
    """A single sample from the ReCast benchmark."""

    id: int
    title: str
    source: str
    url: str
    domains: list[str]
    num_nodes: int
    num_edges: int
    explicitness: float
    nodes: list[str]
    edges: list[dict[str, str]]
    node_explicitness: dict[str, int]
    input_text: str
    abstract: str
    publication_date: str

    @classmethod
    def from_hf_row(cls, row: dict) -> "ReCastSample":
        """Create a ReCastSample from a HuggingFace dataset row."""
        return cls(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            url=row["url"],
            domains=json.loads(row["domains"]) if isinstance(row["domains"], str) else row["domains"],
            num_nodes=row["num_nodes"],
            num_edges=row["num_edges"],
            explicitness=row["explicitness"],
            nodes=json.loads(row["nodes"]) if isinstance(row["nodes"], str) else row["nodes"],
            edges=json.loads(row["edges"]) if isinstance(row["edges"], str) else row["edges"],
            node_explicitness=json.loads(row["node_explicitness"]) if isinstance(row["node_explicitness"], str) else row["node_explicitness"],
            input_text=row["input_text"],
            abstract=row["abstract"],
            publication_date=row["publication_date"],
        )


class ReCastDataset:
    """Container for ReCast benchmark samples."""

    def __init__(self, samples: list[ReCastSample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[ReCastSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> ReCastSample:
        return self._samples[idx]

    def filter_by_domain(self, domain: str) -> "ReCastDataset":
        """Filter to samples containing the specified domain."""
        return ReCastDataset([s for s in self._samples if domain in s.domains])

    def filter_by_domains(self, domains: list[str], require_all: bool = False) -> "ReCastDataset":
        """Filter to samples containing any/all of the specified domains."""
        if require_all:
            return ReCastDataset([s for s in self._samples if all(d in s.domains for d in domains)])
        return ReCastDataset([s for s in self._samples if any(d in s.domains for d in domains)])

    def filter_by_source(self, source: str) -> "ReCastDataset":
        """Filter to samples from a specific journal/source."""
        return ReCastDataset([s for s in self._samples if s.source == source])

    def filter_by_min_explicitness(self, min_val: float) -> "ReCastDataset":
        """Filter to samples with explicitness >= min_val."""
        return ReCastDataset([s for s in self._samples if s.explicitness >= min_val])

    def filter_by_max_explicitness(self, max_val: float) -> "ReCastDataset":
        """Filter to samples with explicitness <= max_val."""
        return ReCastDataset([s for s in self._samples if s.explicitness <= max_val])

    def filter_by_date_before(self, date: str) -> "ReCastDataset":
        """Filter to samples published before a date (YYYY-MM-DD format)."""
        return ReCastDataset([s for s in self._samples if s.publication_date < date])

    def filter_by_date_after(self, date: str) -> "ReCastDataset":
        """Filter to samples published on or after a date (YYYY-MM-DD format)."""
        return ReCastDataset([s for s in self._samples if s.publication_date >= date])

    def filter_by_date_range(self, start: str, end: str) -> "ReCastDataset":
        """Filter to samples published within a date range (inclusive)."""
        return ReCastDataset([s for s in self._samples if start <= s.publication_date <= end])

    def filter_by_min_nodes(self, min_nodes: int) -> "ReCastDataset":
        """Filter to samples with at least min_nodes nodes."""
        return ReCastDataset([s for s in self._samples if s.num_nodes >= min_nodes])

    def filter_by_max_nodes(self, max_nodes: int) -> "ReCastDataset":
        """Filter to samples with at most max_nodes nodes."""
        return ReCastDataset([s for s in self._samples if s.num_nodes <= max_nodes])

    def filter_by_min_edges(self, min_edges: int) -> "ReCastDataset":
        """Filter to samples with at least min_edges edges."""
        return ReCastDataset([s for s in self._samples if s.num_edges >= min_edges])

    def filter_by_max_edges(self, max_edges: int) -> "ReCastDataset":
        """Filter to samples with at most max_edges edges."""
        return ReCastDataset([s for s in self._samples if s.num_edges <= max_edges])

    def filter_by_ids(self, ids: list[int]) -> "ReCastDataset":
        """Filter to samples with specific IDs."""
        id_set = set(ids)
        return ReCastDataset([s for s in self._samples if s.id in id_set])

    def statistics(self) -> dict:
        """Compute statistics about the dataset.

        Returns:
            Dictionary with statistics including:
            - total_samples: Number of samples
            - domains: Distribution of domains
            - sources: Distribution of journal sources
            - publication_years: Distribution by year
            - nodes: Min/max/mean node count
            - edges: Min/max/mean edge count
            - explicitness: Min/max/mean explicitness score
        """
        from collections import Counter
        from statistics import mean

        if not self._samples:
            return {"total_samples": 0}

        # Domain distribution
        domain_counts: Counter[str] = Counter()
        for s in self._samples:
            for d in s.domains:
                domain_counts[d] += 1

        # Source distribution
        source_counts = Counter(s.source for s in self._samples)

        # Year distribution (only samples with dates)
        dated_samples = [s for s in self._samples if s.publication_date]
        year_counts = Counter(s.publication_date[:4] for s in dated_samples)

        # Node/edge statistics
        node_counts = [s.num_nodes for s in self._samples]
        edge_counts = [s.num_edges for s in self._samples]
        explicitness_vals = [s.explicitness for s in self._samples]

        # Date range (handle missing dates)
        if dated_samples:
            dates = [s.publication_date for s in dated_samples]
            date_range = {
                "earliest": min(dates)[:10],  # Trim to YYYY-MM-DD
                "latest": max(dates)[:10],
                "samples_with_dates": len(dated_samples),
            }
        else:
            date_range = {"earliest": "N/A", "latest": "N/A", "samples_with_dates": 0}

        return {
            "total_samples": len(self._samples),
            "domains": dict(domain_counts.most_common()),
            "sources": dict(source_counts.most_common()),
            "publication_years": dict(sorted(year_counts.items())),
            "date_range": date_range,
            "nodes": {
                "min": min(node_counts),
                "max": max(node_counts),
                "mean": round(mean(node_counts), 2),
            },
            "edges": {
                "min": min(edge_counts),
                "max": max(edge_counts),
                "mean": round(mean(edge_counts), 2),
            },
            "explicitness": {
                "min": round(min(explicitness_vals), 3),
                "max": round(max(explicitness_vals), 3),
                "mean": round(mean(explicitness_vals), 3),
            },
        }

    def get_sample_ids(self) -> list[int]:
        """Get list of all sample IDs."""
        return [s.id for s in self._samples]

    def get_domains(self) -> list[str]:
        """Get list of all unique domains."""
        domains: set[str] = set()
        for s in self._samples:
            domains.update(s.domains)
        return sorted(domains)

    def get_sources(self) -> list[str]:
        """Get list of all unique sources (journals)."""
        return sorted(set(s.source for s in self._samples))


def load_dataset(repo: str = "RyanSaklad/ReCast") -> ReCastDataset:
    """Load the ReCast benchmark dataset from HuggingFace.

    Args:
        repo: HuggingFace repository ID

    Returns:
        ReCastDataset containing all 292 benchmark samples

    Example:
        >>> from recast import load_dataset
        >>> ds = load_dataset()
        >>> len(ds)
        292
        >>> sample = ds[0]
        >>> sample.num_nodes
        13
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "Install the 'datasets' package: pip install datasets"
        )

    hf_ds = hf_load(repo, split="test")
    samples = [ReCastSample.from_hf_row(row) for row in hf_ds]
    return ReCastDataset(samples)


@dataclass
class ModelResponse:
    """A model's response to a ReCast sample."""

    id: int
    sample_id: int
    model: str
    task_type: str
    response_answer: str
    response_reasoning: str
    corrected_answer: str
    valid_format: bool
    response_date: str


@dataclass
class Evaluation:
    """An evaluation of a model response."""

    id: int
    response_id: int
    sample_id: int
    model: str
    task_type: str
    evaluator_type: str
    score: float
    evaluation_answer: str
    evaluation_reasoning: str
    evaluation_date: str

    @property
    def parsed_scores(self) -> dict | None:
        """Parse scores from LLM judge evaluation answer (JSON format)."""
        if not self.evaluation_answer:
            return None
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', self.evaluation_answer, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                data = json.loads(self.evaluation_answer)
            return data.get("scores", data)
        except (json.JSONDecodeError, TypeError):
            return None

    @property
    def parsed_fine_grained(self) -> dict | None:
        """Parse fine-grained evaluation answer (YAML format)."""
        if not self.evaluation_answer or self.evaluator_type != "fine_grained":
            return None
        try:
            # Extract YAML from code block if present
            match = re.search(r'```yaml\s*(.*?)\s*```', self.evaluation_answer, re.DOTALL)
            if match:
                yaml_str = match.group(1)
            else:
                yaml_str = self.evaluation_answer
            return yaml.safe_load(yaml_str)
        except (yaml.YAMLError, TypeError):
            return None


def load_responses(repo: str = "RyanSaklad/ReCast") -> list[ModelResponse]:
    """Load pre-computed model responses from HuggingFace.

    These are the original responses used in the paper.

    Returns:
        List of ModelResponse objects
    """
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Install required packages: pip install huggingface_hub pandas"
        )

    path = hf_hub_download(repo, "responses.parquet", repo_type="dataset")
    df = pd.read_parquet(path)

    return [
        ModelResponse(
            id=row["id"],
            sample_id=row["sample_id"],
            model=row["model"],
            task_type=row["task_type"],
            response_answer=row["response_answer"],
            response_reasoning=row["response_reasoning"] or "",
            corrected_answer=row["corrected_answer"] or "",
            valid_format=bool(row["valid_format"]),
            response_date=str(row["response_date"]),
        )
        for _, row in df.iterrows()
    ]


def load_evaluations(repo: str = "RyanSaklad/ReCast") -> list[Evaluation]:
    """Load pre-computed evaluations from HuggingFace.

    Includes both deterministic (graph_similarity) and LLM judge evaluations.

    Returns:
        List of Evaluation objects
    """
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Install required packages: pip install huggingface_hub pandas"
        )

    path = hf_hub_download(repo, "evaluations.parquet", repo_type="dataset")
    df = pd.read_parquet(path)

    return [
        Evaluation(
            id=row["id"],
            response_id=row["response_id"],
            sample_id=row["sample_id"],
            model=row["model"],
            task_type=row["task_type"],
            evaluator_type=row["evaluator_type"],
            score=row["score"],
            evaluation_answer=row["evaluation_answer"] or "",
            evaluation_reasoning=row["evaluation_reasoning"] or "",
            evaluation_date=str(row["evaluation_date"]),
        )
        for _, row in df.iterrows()
    ]


def get_paper_results(
    evaluator: str = "llm_judge",
    task_type: str = "causal_graph_generation",
    repo: str = "RyanSaklad/ReCast",
) -> dict[str, dict]:
    """Get aggregated results matching the paper's tables.

    Args:
        evaluator: "llm_judge", "graph_similarity", "fine_grained", or "deterministic"
        task_type: "causal_graph_generation" or "causal_graph_generation_with_node_names"
        repo: HuggingFace repository ID

    Returns:
        Dict mapping model names to their metrics (mean and std)

    Example:
        >>> results = get_paper_results("llm_judge")
        >>> print(results["deepseek/deepseek-r1"])
        {'causal_accuracy': {'mean': 3.19, 'std': 0.82}, ...}

        >>> # For with_node_names, use deterministic to get SHD/precision/recall
        >>> results = get_paper_results("deterministic", "causal_graph_generation_with_node_names")
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Install numpy: pip install numpy")

    # Handle deterministic evaluation (computed from responses)
    if evaluator == "deterministic":
        return _compute_deterministic_metrics(task_type, repo)

    evaluations = load_evaluations(repo)

    # Filter by evaluator type
    if evaluator == "llm_judge":
        evals = [e for e in evaluations if "llm_judge" in e.evaluator_type]
    elif evaluator == "fine_grained":
        evals = [e for e in evaluations if e.evaluator_type == "fine_grained"]
    else:
        evals = [e for e in evaluations if e.evaluator_type == "graph_similarity"]

    # Filter by task type
    evals = [e for e in evals if e.task_type == task_type]

    # Group by model and calculate statistics
    results = {}
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
            # Compute metrics from fine-grained labels
            metrics = _compute_fine_grained_metrics(scores_list)
            results[model] = metrics
        else:
            scores = [s["score"] for s in scores_list]
            results[model] = {
                "score": {"mean": np.mean(scores), "std": np.std(scores)},
                "n": len(scores_list),
            }

    return results


def _get_overall_scores(fg_data: dict) -> dict[str, float]:
    """Compute weighted scores from fine-grained evaluation labels."""
    from statistics import mean

    _PRESENCE = {
        "PRESENCE_STRONG_MATCH": 1.0,
        "PRESENCE_GRAPH_ONLY": 1.0,
        "PRESENCE_EXPLICIT": 1.0,
        "PRESENCE_WEAK_MATCH": 0.5,
        "PRESENCE_IMPLIED": 0.5,
        "PRESENCE_NO_MATCH": 0.0,
    }
    _SEMANTIC = {
        "SEMANTIC_STRONG": 1.0,
        "SEMANTIC_COMPLETE": 1.0,
        "SEMANTIC_MODERATE": 0.5,
        "SEMANTIC_PARTIAL": 0.5,
        "SEMANTIC_WEAK": 0.25,
        "SEMANTIC_MINIMAL": 0.25,
        "SEMANTIC_NA": 0.0,
    }
    _ABSTRACTION = {
        "ABSTRACTION_ALIGNED": 1.0,
        "ABSTRACTION_BROADER": 0.5,
        "ABSTRACTION_NARROWER": 0.5,
        "ABSTRACTION_NA": 0.0,
    }
    _DIRECTIONALITY = {
        "DIRECTION_CORRECT": 1.0,
        "DIRECTION_UNCLEAR": 0.5,
        "DIRECTION_REVERSED": 0.25,
        "DIRECTION_MISSING": 0.0,
        "DIRECTION_NA": 0.0,
    }
    _INFERENCE = {
        "INFERENCE_DIRECT": 1.0,
        "INFERENCE_DERIVED": 1.0,
        "INFERENCE_STRETCHED": 0.25,
        "INFERENCE_NA": 0.0,
    }
    _IMPORTANCE = {
        "IMPORTANCE_CORE": 1.0,
        "IMPORTANCE_CENTRAL": 1.0,
        "IMPORTANCE_INTERMEDIATE": 0.5,
        "IMPORTANCE_CONNECTING": 0.5,
        "IMPORTANCE_PERIPHERAL": 0.25,
        "IMPORTANCE_AUXILIARY": 0.25,
    }

    def _score(table: dict[str, float], value: str | None) -> float:
        return table.get(value or "", 0.0)

    def composite(block: dict) -> float:
        has_dir = "directionality_label" in block
        has_inf = "inference_label" in block

        if not has_dir and not has_inf:
            # Node precision/recall: presence, semantic, abstraction
            vals = (
                _score(_PRESENCE, block.get("presence_label")),
                _score(_SEMANTIC, block.get("semantic_label")),
                _score(_ABSTRACTION, block.get("abstraction_label")),
            )
        elif has_inf:
            # Edge precision text: presence, inference, abstraction
            vals = (
                _score(_PRESENCE, block.get("presence_label")),
                _score(_INFERENCE, block.get("inference_label")),
                _score(_ABSTRACTION, block.get("abstraction_label")),
            )
        else:
            # Edge precision graph / edge recall: presence, directionality, abstraction
            vals = (
                _score(_PRESENCE, block.get("presence_label")),
                _score(_DIRECTIONALITY, block.get("directionality_label")),
                _score(_ABSTRACTION, block.get("abstraction_label")),
            )
        return sum(vals) / len(vals)

    def block_is_no_match(block: dict | None) -> bool:
        return not block or block.get("presence_label") == "PRESENCE_NO_MATCH"

    def precision_item(item: dict) -> float:
        g_blk = item.get("graph_evaluation", {})
        t_blk = item.get("text_evaluation", {})
        if block_is_no_match(g_blk) and block_is_no_match(t_blk):
            return 0.0
        return max(composite(g_blk), composite(t_blk))

    def precision_avg(items: list[dict]) -> float:
        return mean(precision_item(i) for i in items) if items else 0.0

    def weighted_recall(item: dict) -> tuple[float, float]:
        comp = 0.0 if item.get("presence_label") == "PRESENCE_NO_MATCH" else composite(item)
        imp = _score(_IMPORTANCE, item.get("importance_label"))
        return comp, imp

    def recall_weighted(items: list[dict]) -> float:
        if not items:
            return 0.0
        num = den = 0.0
        for itm in items:
            s, w = weighted_recall(itm)
            num += s * w
            den += w
        return num / den if den else 0.0

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) else 0.0

    node_p_items = fg_data.get("node_precision_evaluations", [])
    edge_p_items = fg_data.get("edge_precision_evaluations", [])
    node_r_items = fg_data.get("node_recall_evaluations", [])
    edge_r_items = fg_data.get("edge_recall_evaluations", [])

    node_precision = precision_avg(node_p_items)
    edge_precision = precision_avg(edge_p_items)
    node_recall = recall_weighted(node_r_items)
    edge_recall = recall_weighted(edge_r_items)
    node_f1 = f1(node_precision, node_recall)
    edge_f1 = f1(edge_precision, edge_recall)

    prec_den = len(node_p_items) + len(edge_p_items)
    rec_den = len(node_r_items) + len(edge_r_items)

    overall_precision = (
        (node_precision * len(node_p_items) + edge_precision * len(edge_p_items)) / prec_den
        if prec_den else 0.0
    )
    overall_recall = (
        (node_recall * len(node_r_items) + edge_recall * len(edge_r_items)) / rec_den
        if rec_den else 0.0
    )
    overall_f1 = f1(overall_precision, overall_recall)

    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
    }


def _compute_fine_grained_metrics(scores_list: list[dict]) -> dict:
    """Aggregate fine-grained evaluation scores across samples."""
    import numpy as np

    node_precisions = []
    node_recalls = []
    edge_precisions = []
    edge_recalls = []
    f1_scores = []

    for item in scores_list:
        fg = item.get("fine_grained")
        if not fg:
            continue
        try:
            scores = _get_overall_scores(fg)
            node_precisions.append(scores["node_precision"])
            node_recalls.append(scores["node_recall"])
            edge_precisions.append(scores["edge_precision"])
            edge_recalls.append(scores["edge_recall"])
            f1_scores.append(scores["overall_f1"])
        except (KeyError, TypeError, ZeroDivisionError):
            continue

    def stats(vals: list[float]) -> dict:
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": 0.0, "std": 0.0}

    return {
        "node_precision": stats(node_precisions),
        "node_recall": stats(node_recalls),
        "edge_precision": stats(edge_precisions),
        "edge_recall": stats(edge_recalls),
        "f1": stats(f1_scores),
        "n": len(scores_list),
    }


def _compute_deterministic_metrics(
    task_type: str,
    repo: str = "RyanSaklad/ReCast",
) -> dict[str, dict]:
    """Compute deterministic metrics (SHD, precision, recall) from stored responses."""
    import numpy as np
    from .helpers import calculate_full_metrics, nodes_to_node_dict

    responses = load_responses(repo)
    dataset = load_dataset(repo)
    sample_lookup = {s.id: s for s in dataset}
    filtered_responses = [r for r in responses if r.task_type == task_type]
    model_metrics: dict[str, dict[str, list[float]]] = {}

    for resp in filtered_responses:
        if resp.sample_id not in sample_lookup:
            continue

        sample = sample_lookup[resp.sample_id]
        gt_edges = sample.edges
        answer = resp.corrected_answer or resp.response_answer
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

            if resp.model not in model_metrics:
                model_metrics[resp.model] = {k: [] for k in metrics.keys()}

            for k, v in metrics.items():
                model_metrics[resp.model][k].append(v)

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    def stats(vals: list) -> dict:
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    results: dict[str, dict] = {}
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


def load_responses_from_db(db_path: str) -> list[ModelResponse]:
    """Load model responses from a local SQLite database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        List of ModelResponse objects
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, sample_id, model, task_type,
               response_answer, response_reasoning,
               COALESCE(corrected_answer, '') as corrected_answer,
               valid_format, response_date
        FROM benchmark_responses
    """)

    responses = []
    for row in cursor.fetchall():
        responses.append(ModelResponse(
            id=row[0],
            sample_id=row[1],
            model=row[2],
            task_type=row[3],
            response_answer=row[4] or "",
            response_reasoning=row[5] or "",
            corrected_answer=row[6] or "",
            valid_format=bool(row[7]),
            response_date=str(row[8]) if row[8] else "",
        ))

    conn.close()
    return responses


def load_evaluations_from_db(db_path: str) -> list[Evaluation]:
    """Load evaluations from a local SQLite database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        List of Evaluation objects
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.id, e.response_id, r.sample_id, r.model, r.task_type,
               e.evaluator_type, e.score, e.evaluation_answer, e.evaluation_reasoning,
               e.evaluation_date
        FROM benchmark_evaluations e
        JOIN benchmark_responses r ON e.response_id = r.id
    """)

    evaluations = []
    for row in cursor.fetchall():
        evaluations.append(Evaluation(
            id=row[0],
            response_id=row[1],
            sample_id=row[2],
            model=row[3],
            task_type=row[4],
            evaluator_type=row[5],
            score=row[6] if row[6] is not None else 0.0,
            evaluation_answer=row[7] or "",
            evaluation_reasoning=row[8] or "",
            evaluation_date=str(row[9]) if row[9] else "",
        ))

    conn.close()
    return evaluations


def export_to_parquet(
    db_path: str,
    output_dir: str | Path = ".",
    include_responses: bool = True,
    include_evaluations: bool = True,
) -> dict[str, Path]:
    """Export local database results to parquet files.

    Creates parquet files compatible with the HuggingFace dataset format.

    Args:
        db_path: Path to the SQLite database
        output_dir: Directory to write parquet files
        include_responses: Whether to export responses
        include_evaluations: Whether to export evaluations

    Returns:
        Dictionary mapping file type to output path

    Example:
        >>> paths = export_to_parquet("recast_results.db", "exports/")
        >>> print(paths)
        {'responses': Path('exports/responses.parquet'), 'evaluations': Path('exports/evaluations.parquet')}
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas: pip install pandas")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    if include_responses:
        responses = load_responses_from_db(db_path)
        df = pd.DataFrame([
            {
                "id": r.id,
                "sample_id": r.sample_id,
                "model": r.model,
                "task_type": r.task_type,
                "response_answer": r.response_answer,
                "response_reasoning": r.response_reasoning,
                "corrected_answer": r.corrected_answer,
                "valid_format": int(r.valid_format),
                "response_date": r.response_date,
            }
            for r in responses
        ])
        path = output_dir / "responses.parquet"
        df.to_parquet(path, index=False)
        exported["responses"] = path

    if include_evaluations:
        evaluations = load_evaluations_from_db(db_path)
        df = pd.DataFrame([
            {
                "id": e.id,
                "response_id": e.response_id,
                "sample_id": e.sample_id,
                "model": e.model,
                "task_type": e.task_type,
                "evaluator_type": e.evaluator_type,
                "score": e.score,
                "evaluation_answer": e.evaluation_answer,
                "evaluation_reasoning": e.evaluation_reasoning,
                "evaluation_date": e.evaluation_date,
            }
            for e in evaluations
        ])
        path = output_dir / "evaluations.parquet"
        df.to_parquet(path, index=False)
        exported["evaluations"] = path

    return exported
