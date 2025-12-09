"""Helper utilities for ReCast benchmark."""

from __future__ import annotations

import json
import os
import random
import sqlite3
import warnings
from pathlib import Path
from typing import TypedDict

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate

# Suppress CDT warnings before import
warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ.setdefault("CDT_VERBOSE", "0")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cdt.metrics import precision_recall as cdt_precision_recall, SHD as cdt_SHD


class EdgeDict(TypedDict, total=False):
    source: str
    sink: str
    target: str  # Alias for sink


class NodeEntry(TypedDict):
    name: str
    id: int


class MetricsResult(TypedDict):
    precision: float
    recall: float
    f1: float
    shd: float
    normalized_shd: float
    sample_count: int


class MetricsWithStd(TypedDict):
    mean: float
    std: float


class DetailedMetricsResult(TypedDict):
    precision: MetricsWithStd
    recall: MetricsWithStd
    f1: MetricsWithStd
    shd: MetricsWithStd
    normalized_shd: MetricsWithStd
    sample_count: int


class FullMetricsResult(TypedDict):
    """Full metrics result with node and edge level metrics."""
    node_precision: MetricsWithStd
    node_recall: MetricsWithStd
    edge_precision: MetricsWithStd
    edge_recall: MetricsWithStd
    f1: MetricsWithStd
    shd: MetricsWithStd
    normalized_shd: MetricsWithStd
    sample_count: int


class SampleMetrics(TypedDict):
    """Metrics for a single sample."""
    node_precision: float
    node_recall: float
    edge_precision: float
    edge_recall: float
    f1: float
    shd: float
    normalized_shd: float


def nodes_to_node_dict(nodes: set[str]) -> dict[str, list[NodeEntry]]:
    """Convert node names to a dictionary with sequential IDs starting from 1."""
    return {
        "nodes": [
            {"name": node, "id": i + 1}
            for i, node in enumerate(sorted(nodes))
        ]
    }


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template by name from the prompts/ directory."""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"

    for ext in [".md", ".txt"]:
        prompt_path = prompts_dir / f"{prompt_name}{ext}"
        if prompt_path.exists():
            return prompt_path.read_text()

    raise FileNotFoundError(f"Prompt '{prompt_name}' not found in {prompts_dir}")


def create_db(db_path: str = "recast_results.db") -> None:
    """Create the database schema for storing benchmark results.

    Column names match HuggingFace dataset schema for easy export.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Causal graphs table (stores ground truth)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS causal_graphs (
        id INTEGER PRIMARY KEY,
        sample_id INTEGER,
        domains TEXT,
        corrected_json TEXT,
        input_text TEXT
    )
    ''')

    # Benchmark responses table (matches HuggingFace responses.parquet)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmark_responses (
        id INTEGER PRIMARY KEY,
        sample_id INTEGER,
        response_date TEXT,
        task_type TEXT,
        model TEXT,
        response_reasoning TEXT,
        response_answer TEXT,
        finish_reason TEXT,
        valid_format INTEGER,
        corrected_reasoning TEXT,
        corrected_answer TEXT,
        FOREIGN KEY (sample_id) REFERENCES causal_graphs (sample_id)
    )
    ''')

    # Benchmark evaluations table (matches HuggingFace evaluations.parquet)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmark_evaluations (
        id INTEGER PRIMARY KEY,
        response_id INTEGER,
        evaluation_date TEXT,
        evaluator_type TEXT,
        score REAL,
        evaluation_reasoning TEXT,
        evaluation_answer TEXT,
        FOREIGN KEY (response_id) REFERENCES benchmark_responses (id)
    )
    ''')

    conn.commit()
    conn.close()


def calculate_graph_metrics(
    ground_truth: list[EdgeDict],
    generated: list[EdgeDict],
) -> tuple[float, float, float, float, float]:
    """Calculate precision, recall, F1, SHD, and normalized SHD for causal graphs."""
    gt_graph = nx.DiGraph()
    gen_graph = nx.DiGraph()

    for rel in ground_truth:
        target = rel.get("sink") or rel.get("target")
        gt_graph.add_edge(rel["source"], target)

    for rel in generated:
        target = rel.get("sink") or rel.get("target")
        gen_graph.add_edge(rel["source"], target)

    all_nodes = sorted(set(gt_graph.nodes()) | set(gen_graph.nodes()))
    n_nodes = len(all_nodes)

    gt_matrix: NDArray[np.float32] = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    gen_matrix: NDArray[np.float32] = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    for u, v in gt_graph.edges():
        gt_matrix[node_to_idx[u], node_to_idx[v]] = 1.0

    for u, v in gen_graph.edges():
        gen_matrix[node_to_idx[u], node_to_idx[v]] = 1.0

    precision, recall = cdt_precision_recall(gt_matrix, gen_matrix)
    if isinstance(precision, (list, np.ndarray)):
        precision = float(np.mean(precision))
    if isinstance(recall, (list, np.ndarray)):
        recall = float(np.mean(recall))
    precision = float(precision)
    recall = float(recall)

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    shd_value = cdt_SHD(gt_matrix, gen_matrix)
    if isinstance(shd_value, (list, np.ndarray)):
        shd_value = float(np.mean(shd_value))
    shd_value = float(shd_value)

    max_edges = n_nodes * (n_nodes - 1)
    normalized_shd = shd_value / max_edges if max_edges > 0 else 0.0

    return precision, recall, f1_score, shd_value, normalized_shd


def generate_random_baseline(
    ground_truth: list[EdgeDict],
    n_samples: int = 100,
    best_k: int | None = None,
) -> dict[str, MetricsResult]:
    """Generate random baseline metrics using random graphs with same nodes as ground truth.

    Args:
        ground_truth: List of ground truth edges
        n_samples: Number of random samples to generate
        best_k: If provided, also returns metrics for the best k samples by F1

    Returns:
        Dictionary with 'average' metrics, and optionally 'best_k' metrics
    """
    all_nodes: list[str] = sorted({
        node for rel in ground_truth
        for node in (rel["source"], rel.get("sink") or rel.get("target"))
    })

    metrics: dict[str, list[float]] = {
        'precision': [], 'recall': [], 'f1': [], 'shd': [], 'normalized_shd': []
    }

    n_edges = len(ground_truth)

    for _ in range(n_samples):
        random_edges: list[EdgeDict] = []
        existing_edges: set[tuple[str, str]] = set()

        while len(random_edges) < n_edges:
            source = random.choice(all_nodes)
            sink = random.choice(all_nodes)
            edge = (source, sink)

            if source != sink and edge not in existing_edges:
                random_edges.append({"source": source, "sink": sink})
                existing_edges.add(edge)

        p, r, f1, shd, nshd = calculate_graph_metrics(ground_truth, random_edges)
        metrics['precision'].append(p)
        metrics['recall'].append(r)
        metrics['f1'].append(f1)
        metrics['shd'].append(shd)
        metrics['normalized_shd'].append(nshd)

    result: dict[str, MetricsResult] = {
        'average': {
            'precision': sum(metrics['precision']) / n_samples,
            'recall': sum(metrics['recall']) / n_samples,
            'f1': sum(metrics['f1']) / n_samples,
            'shd': sum(metrics['shd']) / n_samples,
            'normalized_shd': sum(metrics['normalized_shd']) / n_samples,
            'sample_count': n_samples,
        }
    }

    if best_k is not None:
        sorted_indices = sorted(
            range(len(metrics['f1'])),
            key=lambda i: metrics['f1'][i],
            reverse=True
        )[:best_k]

        result['best_k'] = {
            'precision': sum(metrics['precision'][i] for i in sorted_indices) / best_k,
            'recall': sum(metrics['recall'][i] for i in sorted_indices) / best_k,
            'f1': sum(metrics['f1'][i] for i in sorted_indices) / best_k,
            'shd': sum(metrics['shd'][i] for i in sorted_indices) / best_k,
            'normalized_shd': sum(metrics['normalized_shd'][i] for i in sorted_indices) / best_k,
            'sample_count': n_samples,
        }

    return result


def get_model_metrics(db_path: str = "recast_results.db") -> dict[str, DetailedMetricsResult]:
    """Calculate precision, recall, F1 scores, and SHD for each model's responses."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT br.model, COALESCE(br.corrected_answer, br.response_answer) as answer, cg.corrected_json
        FROM benchmark_responses br
        JOIN causal_graphs cg ON br.sample_id = cg.sample_id
        WHERE cg.corrected_json IS NOT NULL
    ''')

    results = cursor.fetchall()
    conn.close()

    model_metrics: dict[str, dict[str, list[float]]] = {}

    for model, answer, ground_truth in results:
        if not answer or not ground_truth:
            continue

        try:
            if "```json" in answer:
                answer = answer.split("```json")[1].split("```")[0]
            elif "```" in answer:
                answer = answer.split("```")[1].split("```")[0]

            generated_data = json.loads(answer)
            ground_truth_data = json.loads(ground_truth)

            if isinstance(generated_data, list):
                generated_data = {"relationships": generated_data}

            precision, recall, f1, shd, normalized_shd = calculate_graph_metrics(
                ground_truth_data,
                generated_data.get("relationships", [])
            )

            if model not in model_metrics:
                model_metrics[model] = {
                    'precision': [], 'recall': [], 'f1': [],
                    'shd': [], 'normalized_shd': []
                }

            model_metrics[model]['precision'].append(precision)
            model_metrics[model]['recall'].append(recall)
            model_metrics[model]['f1'].append(f1)
            model_metrics[model]['shd'].append(shd)
            model_metrics[model]['normalized_shd'].append(normalized_shd)

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    final_metrics: dict[str, DetailedMetricsResult] = {}
    for model, metrics in model_metrics.items():
        final_metrics[model] = {
            'precision': {
                'mean': float(np.mean(metrics['precision'])),
                'std': float(np.std(metrics['precision']))
            },
            'recall': {
                'mean': float(np.mean(metrics['recall'])),
                'std': float(np.std(metrics['recall']))
            },
            'f1': {
                'mean': float(np.mean(metrics['f1'])),
                'std': float(np.std(metrics['f1']))
            },
            'shd': {
                'mean': float(np.mean(metrics['shd'])),
                'std': float(np.std(metrics['shd']))
            },
            'normalized_shd': {
                'mean': float(np.mean(metrics['normalized_shd'])),
                'std': float(np.std(metrics['normalized_shd']))
            },
            'sample_count': len(metrics['precision'])
        }

    return final_metrics


def calculate_full_metrics(
    ground_truth: list[EdgeDict],
    generated: list[EdgeDict],
) -> SampleMetrics:
    """Calculate node-level and edge-level precision/recall, F1, and SHD."""
    gt_nodes: set[str] = set()
    for rel in ground_truth:
        gt_nodes.add(rel["source"])
        target = rel.get("sink") or rel.get("target")
        if target:
            gt_nodes.add(target)

    gen_nodes: set[str] = set()
    for rel in generated:
        gen_nodes.add(rel["source"])
        target = rel.get("sink") or rel.get("target")
        if target:
            gen_nodes.add(target)

    node_tp = len(gt_nodes & gen_nodes)
    node_precision = node_tp / len(gen_nodes) if gen_nodes else 0.0
    node_recall = node_tp / len(gt_nodes) if gt_nodes else 0.0

    edge_precision, edge_recall, f1, shd, normalized_shd = calculate_graph_metrics(
        ground_truth, generated
    )

    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "f1": f1,
        "shd": shd,
        "normalized_shd": normalized_shd,
    }


def get_full_model_metrics(
    db_path: str = "recast_results.db",
    task_type: str | None = None,
) -> dict[str, FullMetricsResult]:
    """Calculate full metrics (node + edge level) for each model's responses."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Try new schema first, fall back to legacy schema
    try:
        query = '''
            SELECT br.model, COALESCE(br.corrected_answer, br.response_answer) as answer,
                   cg.corrected_json, br.task_type
            FROM benchmark_responses br
            JOIN causal_graphs cg ON br.sample_id = cg.sample_id
            WHERE cg.corrected_json IS NOT NULL
        '''
        params: tuple[str, ...] = ()
        if task_type:
            query += ' AND br.task_type = ?'
            params = (task_type,)
        cursor.execute(query, params)
    except sqlite3.OperationalError:
        # Legacy schema: causal_graph_id, evaluation_type
        query = '''
            SELECT br.model, COALESCE(br.corrected_answer, br.response_answer) as answer,
                   cg.corrected_json, br.evaluation_type
            FROM benchmark_responses br
            JOIN causal_graphs cg ON br.causal_graph_id = cg.id
            WHERE cg.corrected_json IS NOT NULL
        '''
        params = ()
        if task_type:
            query += ' AND br.evaluation_type = ?'
            params = (task_type,)
        cursor.execute(query, params)

    results = cursor.fetchall()
    conn.close()

    model_metrics: dict[str, dict[str, list[float]]] = {}

    for model, answer, ground_truth, row_task_type in results:
        if not answer or not ground_truth:
            continue

        try:
            if "```json" in answer:
                answer = answer.split("```json")[1].split("```")[0]
            elif "```" in answer:
                answer = answer.split("```")[1].split("```")[0]

            generated_data = json.loads(answer)
            ground_truth_data = json.loads(ground_truth)

            if isinstance(generated_data, list):
                generated_data = {"relationships": generated_data}

            if row_task_type == "causal_graph_generation_with_node_names":
                gt_variables: set[str] = set()
                for rel in ground_truth_data:
                    gt_variables.add(rel["source"])
                    gt_variables.add(rel.get("sink") or rel.get("target"))

                node_dict = nodes_to_node_dict(gt_variables)
                id_to_name = {node["id"]: node["name"] for node in node_dict["nodes"]}

                for rel in generated_data.get("relationships", []):
                    if isinstance(rel.get("source"), int):
                        rel["source"] = id_to_name.get(rel["source"], str(rel["source"]))
                    if isinstance(rel.get("sink"), int):
                        rel["sink"] = id_to_name.get(rel["sink"], str(rel["sink"]))

            metrics = calculate_full_metrics(
                ground_truth_data,
                generated_data.get("relationships", [])
            )

            if model not in model_metrics:
                model_metrics[model] = {
                    'node_precision': [], 'node_recall': [],
                    'edge_precision': [], 'edge_recall': [],
                    'f1': [], 'shd': [], 'normalized_shd': []
                }

            model_metrics[model]['node_precision'].append(metrics['node_precision'])
            model_metrics[model]['node_recall'].append(metrics['node_recall'])
            model_metrics[model]['edge_precision'].append(metrics['edge_precision'])
            model_metrics[model]['edge_recall'].append(metrics['edge_recall'])
            model_metrics[model]['f1'].append(metrics['f1'])
            model_metrics[model]['shd'].append(metrics['shd'])
            model_metrics[model]['normalized_shd'].append(metrics['normalized_shd'])

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    final_metrics: dict[str, FullMetricsResult] = {}
    for model, metrics in model_metrics.items():
        final_metrics[model] = {
            'node_precision': {
                'mean': float(np.mean(metrics['node_precision'])),
                'std': float(np.std(metrics['node_precision']))
            },
            'node_recall': {
                'mean': float(np.mean(metrics['node_recall'])),
                'std': float(np.std(metrics['node_recall']))
            },
            'edge_precision': {
                'mean': float(np.mean(metrics['edge_precision'])),
                'std': float(np.std(metrics['edge_precision']))
            },
            'edge_recall': {
                'mean': float(np.mean(metrics['edge_recall'])),
                'std': float(np.std(metrics['edge_recall']))
            },
            'f1': {
                'mean': float(np.mean(metrics['f1'])),
                'std': float(np.std(metrics['f1']))
            },
            'shd': {
                'mean': float(np.mean(metrics['shd'])),
                'std': float(np.std(metrics['shd']))
            },
            'normalized_shd': {
                'mean': float(np.mean(metrics['normalized_shd'])),
                'std': float(np.std(metrics['normalized_shd']))
            },
            'sample_count': len(metrics['node_precision'])
        }

    return final_metrics


def format_metrics_table(
    metrics: dict[str, FullMetricsResult],
    tablefmt: str = "rounded_outline",
) -> str:
    """Format metrics as a table with mean±std values."""
    headers = ["Model", "Node Prec", "Node Rec", "Edge Prec", "Edge Rec", "F1", "SHD", "Norm SHD", "N"]
    rows = []

    for model, m in sorted(metrics.items()):
        rows.append([
            model,
            f"{m['node_precision']['mean']:.3f}±{m['node_precision']['std']:.3f}",
            f"{m['node_recall']['mean']:.3f}±{m['node_recall']['std']:.3f}",
            f"{m['edge_precision']['mean']:.3f}±{m['edge_precision']['std']:.3f}",
            f"{m['edge_recall']['mean']:.3f}±{m['edge_recall']['std']:.3f}",
            f"{m['f1']['mean']:.3f}±{m['f1']['std']:.3f}",
            f"{m['shd']['mean']:.1f}±{m['shd']['std']:.1f}",
            f"{m['normalized_shd']['mean']:.3f}±{m['normalized_shd']['std']:.3f}",
            m['sample_count'],
        ])

    return tabulate(rows, headers=headers, tablefmt=tablefmt)
