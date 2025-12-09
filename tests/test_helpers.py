"""Tests for the ReCast helpers module."""

import os
import sqlite3
import tempfile

import pytest

from recast import calculate_graph_metrics, calculate_full_metrics, load_prompt
from recast.helpers import (
    nodes_to_node_dict,
    create_db,
    generate_random_baseline,
)
from recast.dataset import _get_overall_scores


class TestCalculateGraphMetrics:
    """Test graph metrics calculation.

    Note: These tests use CDT's precision_recall which computes area under
    precision-recall curve, not simple set-based precision/recall.
    """

    def test_shd_zero_for_perfect_match(self):
        """Perfect prediction should have SHD=0."""
        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        p, r, f1, shd, nshd = calculate_graph_metrics(edges, edges)
        assert shd == 0.0
        assert nshd == 0.0

    def test_shd_counts_differences(self):
        """SHD should count edge differences."""
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        pred = [
            {"source": "A", "target": "B"},  # correct
            {"source": "A", "target": "C"},  # wrong (should be B->C)
        ]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert shd == 2.0  # 1 missing (B->C) + 1 extra (A->C)

    def test_normalized_shd_bounded(self):
        """Normalized SHD should be between 0 and 1."""
        gt = [{"source": "A", "target": "B"}]
        pred = [{"source": "C", "target": "D"}]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert 0.0 <= nshd <= 1.0

    def test_f1_calculation(self):
        """F1 should be harmonic mean of precision and recall."""
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "D"},
        ]
        pred = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert abs(f1 - expected_f1) < 1e-10

    def test_metrics_bounded(self):
        """All metrics should be in valid ranges."""
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        pred = [
            {"source": "A", "target": "B"},
            {"source": "C", "target": "D"},
        ]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert shd >= 0.0
        assert 0.0 <= nshd <= 1.0

    def test_accepts_sink_key(self):
        """Should accept 'sink' as alias for 'target'."""
        gt = [{"source": "A", "sink": "B"}]
        pred = [{"source": "A", "sink": "B"}]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert shd == 0.0  # Perfect match

    def test_mixed_sink_and_target(self):
        """Should handle mix of 'sink' and 'target' keys."""
        gt = [{"source": "A", "sink": "B"}]
        pred = [{"source": "A", "target": "B"}]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert shd == 0.0  # Perfect match


class TestNodesToNodeDict:
    """Test node dictionary conversion."""

    def test_empty_set(self):
        result = nodes_to_node_dict(set())
        assert result == {"nodes": []}

    def test_single_node(self):
        result = nodes_to_node_dict({"A"})
        assert result == {"nodes": [{"name": "A", "id": 1}]}

    def test_multiple_nodes_sorted(self):
        result = nodes_to_node_dict({"C", "A", "B"})
        assert result == {
            "nodes": [
                {"name": "A", "id": 1},
                {"name": "B", "id": 2},
                {"name": "C", "id": 3},
            ]
        }

    def test_ids_start_from_one(self):
        result = nodes_to_node_dict({"X", "Y"})
        ids = [n["id"] for n in result["nodes"]]
        assert min(ids) == 1


class TestLoadPrompt:
    """Test prompt loading."""

    def test_load_existing_prompt(self):
        # This assumes causal_graph_generation prompt exists
        prompt = load_prompt("causal_graph_generation")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_load_nonexistent_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")

    def test_prompt_contains_expected_content(self):
        prompt = load_prompt("causal_graph_generation")
        # Should reference nodes/edges in some form
        assert "relationship" in prompt.lower() or "causal" in prompt.lower()


class TestCreateDb:
    """Test database creation."""

    def test_creates_database_file(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            create_db(db_path)
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)

    def test_creates_expected_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            create_db(db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            assert "causal_graphs" in tables
            assert "benchmark_responses" in tables
            assert "benchmark_evaluations" in tables

            conn.close()
        finally:
            os.unlink(db_path)

    def test_idempotent(self):
        """Calling create_db twice should not raise."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            create_db(db_path)
            create_db(db_path)  # Should not raise
        finally:
            os.unlink(db_path)


class TestGenerateRandomBaseline:
    """Test random baseline generation."""

    def test_returns_average_metrics(self):
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        result = generate_random_baseline(gt, n_samples=10)
        assert "average" in result
        assert "precision" in result["average"]
        assert "recall" in result["average"]
        assert "f1" in result["average"]
        assert "shd" in result["average"]

    def test_sample_count_tracked(self):
        gt = [{"source": "A", "target": "B"}]
        result = generate_random_baseline(gt, n_samples=50)
        assert result["average"]["sample_count"] == 50

    def test_best_k_when_requested(self):
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        result = generate_random_baseline(gt, n_samples=20, best_k=5)
        assert "best_k" in result
        # Best k should have better or equal F1 than average
        assert result["best_k"]["f1"] >= result["average"]["f1"]

    def test_metrics_in_valid_range(self):
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "D"},
        ]
        result = generate_random_baseline(gt, n_samples=50)
        avg = result["average"]
        assert 0.0 <= avg["precision"] <= 1.0
        assert 0.0 <= avg["recall"] <= 1.0
        assert 0.0 <= avg["f1"] <= 1.0
        assert avg["shd"] >= 0.0
        assert 0.0 <= avg["normalized_shd"] <= 1.0


class TestCalculateFullMetrics:
    """Test full metrics calculation including node-level metrics."""

    def test_perfect_match_shd_zero(self):
        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        result = calculate_full_metrics(edges, edges)
        assert result["node_precision"] == 1.0
        assert result["node_recall"] == 1.0
        assert result["shd"] == 0.0

    def test_no_node_overlap(self):
        gt = [{"source": "A", "target": "B"}]
        pred = [{"source": "C", "target": "D"}]
        result = calculate_full_metrics(gt, pred)
        assert result["node_precision"] == 0.0
        assert result["node_recall"] == 0.0

    def test_partial_node_match(self):
        gt = [{"source": "A", "target": "B"}]
        pred = [{"source": "A", "target": "C"}]  # A matches, C doesn't
        result = calculate_full_metrics(gt, pred)
        # A is in both, B is only in gt, C is only in pred
        # node_precision = 1/2 (A is correct out of A,C)
        # node_recall = 1/2 (A found out of A,B)
        assert result["node_precision"] == 0.5
        assert result["node_recall"] == 0.5

    def test_returns_all_metrics(self):
        gt = [{"source": "A", "target": "B"}]
        pred = [{"source": "A", "target": "B"}]
        result = calculate_full_metrics(gt, pred)
        expected_keys = ["node_precision", "node_recall", "edge_precision", "edge_recall", "f1", "shd", "normalized_shd"]
        for key in expected_keys:
            assert key in result

    def test_metrics_bounded(self):
        gt = [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}]
        pred = [{"source": "A", "target": "B"}, {"source": "X", "target": "Y"}]
        result = calculate_full_metrics(gt, pred)
        for key in ["node_precision", "node_recall", "edge_precision", "edge_recall", "f1", "normalized_shd"]:
            assert 0.0 <= result[key] <= 1.0
        assert result["shd"] >= 0.0


class TestGetOverallScores:
    """Test fine-grained score calculations."""

    def test_perfect_node_precision(self):
        fg_data = {
            "node_precision_evaluations": [
                {
                    "node_number": 1,
                    "graph_evaluation": {
                        "presence_label": "PRESENCE_STRONG_MATCH",
                        "semantic_label": "SEMANTIC_STRONG",
                        "abstraction_label": "ABSTRACTION_ALIGNED",
                    },
                    "text_evaluation": {
                        "presence_label": "PRESENCE_STRONG_MATCH",
                        "semantic_label": "SEMANTIC_STRONG",
                        "abstraction_label": "ABSTRACTION_ALIGNED",
                    },
                }
            ],
            "node_recall_evaluations": [],
            "edge_precision_evaluations": [],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        assert scores["node_precision"] == 1.0

    def test_no_match_node_precision(self):
        fg_data = {
            "node_precision_evaluations": [
                {
                    "node_number": 1,
                    "graph_evaluation": {
                        "presence_label": "PRESENCE_NO_MATCH",
                        "semantic_label": "SEMANTIC_NA",
                        "abstraction_label": "ABSTRACTION_NA",
                    },
                    "text_evaluation": {
                        "presence_label": "PRESENCE_NO_MATCH",
                        "semantic_label": "SEMANTIC_NA",
                        "abstraction_label": "ABSTRACTION_NA",
                    },
                }
            ],
            "node_recall_evaluations": [],
            "edge_precision_evaluations": [],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        assert scores["node_precision"] == 0.0

    def test_weighted_recall(self):
        fg_data = {
            "node_precision_evaluations": [],
            "node_recall_evaluations": [
                {
                    "node_number": 1,
                    "importance_label": "IMPORTANCE_CORE",
                    "presence_label": "PRESENCE_STRONG_MATCH",
                    "semantic_label": "SEMANTIC_COMPLETE",
                    "abstraction_label": "ABSTRACTION_ALIGNED",
                },
                {
                    "node_number": 2,
                    "importance_label": "IMPORTANCE_PERIPHERAL",
                    "presence_label": "PRESENCE_NO_MATCH",
                    "semantic_label": "SEMANTIC_NA",
                    "abstraction_label": "ABSTRACTION_NA",
                },
            ],
            "edge_precision_evaluations": [],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        # Core node found (weight 1.0), peripheral not found (weight 0.25)
        # Weighted recall should be higher because core node was found
        assert scores["node_recall"] > 0.5

    def test_edge_precision_with_directionality(self):
        fg_data = {
            "node_precision_evaluations": [],
            "node_recall_evaluations": [],
            "edge_precision_evaluations": [
                {
                    "edge_number": 1,
                    "graph_evaluation": {
                        "presence_label": "PRESENCE_STRONG_MATCH",
                        "directionality_label": "DIRECTION_CORRECT",
                        "abstraction_label": "ABSTRACTION_ALIGNED",
                    },
                    "text_evaluation": {
                        "presence_label": "PRESENCE_GRAPH_ONLY",
                        "inference_label": "INFERENCE_DIRECT",
                        "abstraction_label": "ABSTRACTION_ALIGNED",
                    },
                }
            ],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        assert scores["edge_precision"] == 1.0

    def test_f1_calculation(self):
        fg_data = {
            "node_precision_evaluations": [
                {
                    "node_number": 1,
                    "graph_evaluation": {
                        "presence_label": "PRESENCE_STRONG_MATCH",
                        "semantic_label": "SEMANTIC_STRONG",
                        "abstraction_label": "ABSTRACTION_ALIGNED",
                    },
                    "text_evaluation": {
                        "presence_label": "PRESENCE_STRONG_MATCH",
                        "semantic_label": "SEMANTIC_STRONG",
                        "abstraction_label": "ABSTRACTION_ALIGNED",
                    },
                }
            ],
            "node_recall_evaluations": [
                {
                    "node_number": 1,
                    "importance_label": "IMPORTANCE_CORE",
                    "presence_label": "PRESENCE_STRONG_MATCH",
                    "semantic_label": "SEMANTIC_COMPLETE",
                    "abstraction_label": "ABSTRACTION_ALIGNED",
                }
            ],
            "edge_precision_evaluations": [],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        # With perfect precision and recall, F1 should be 1.0
        assert scores["node_f1"] == 1.0

    def test_empty_evaluations(self):
        fg_data = {
            "node_precision_evaluations": [],
            "node_recall_evaluations": [],
            "edge_precision_evaluations": [],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        assert scores["node_precision"] == 0.0
        assert scores["node_recall"] == 0.0
        assert scores["edge_precision"] == 0.0
        assert scores["edge_recall"] == 0.0

    def test_partial_scores(self):
        fg_data = {
            "node_precision_evaluations": [
                {
                    "node_number": 1,
                    "graph_evaluation": {
                        "presence_label": "PRESENCE_WEAK_MATCH",
                        "semantic_label": "SEMANTIC_MODERATE",
                        "abstraction_label": "ABSTRACTION_BROADER",
                    },
                    "text_evaluation": {
                        "presence_label": "PRESENCE_NO_MATCH",
                        "semantic_label": "SEMANTIC_NA",
                        "abstraction_label": "ABSTRACTION_NA",
                    },
                }
            ],
            "node_recall_evaluations": [],
            "edge_precision_evaluations": [],
            "edge_recall_evaluations": [],
        }
        scores = _get_overall_scores(fg_data)
        # Should be between 0 and 1
        assert 0.0 < scores["node_precision"] < 1.0


class TestFormatMetricsTable:
    """Test metrics table formatting."""

    def test_returns_string(self):
        from recast.helpers import format_metrics_table
        metrics = {
            "model_a": {
                "node_precision": {"mean": 0.8, "std": 0.1},
                "node_recall": {"mean": 0.7, "std": 0.15},
                "edge_precision": {"mean": 0.6, "std": 0.2},
                "edge_recall": {"mean": 0.5, "std": 0.25},
                "f1": {"mean": 0.55, "std": 0.18},
                "shd": {"mean": 5.0, "std": 2.0},
                "normalized_shd": {"mean": 0.1, "std": 0.05},
                "sample_count": 100,
            }
        }
        result = format_metrics_table(metrics)
        assert isinstance(result, str)

    def test_contains_model_name(self):
        from recast.helpers import format_metrics_table
        metrics = {
            "test_model": {
                "node_precision": {"mean": 0.8, "std": 0.1},
                "node_recall": {"mean": 0.7, "std": 0.15},
                "edge_precision": {"mean": 0.6, "std": 0.2},
                "edge_recall": {"mean": 0.5, "std": 0.25},
                "f1": {"mean": 0.55, "std": 0.18},
                "shd": {"mean": 5.0, "std": 2.0},
                "normalized_shd": {"mean": 0.1, "std": 0.05},
                "sample_count": 100,
            }
        }
        result = format_metrics_table(metrics)
        assert "test_model" in result

    def test_contains_headers(self):
        from recast.helpers import format_metrics_table
        metrics = {
            "model": {
                "node_precision": {"mean": 0.8, "std": 0.1},
                "node_recall": {"mean": 0.7, "std": 0.15},
                "edge_precision": {"mean": 0.6, "std": 0.2},
                "edge_recall": {"mean": 0.5, "std": 0.25},
                "f1": {"mean": 0.55, "std": 0.18},
                "shd": {"mean": 5.0, "std": 2.0},
                "normalized_shd": {"mean": 0.1, "std": 0.05},
                "sample_count": 100,
            }
        }
        result = format_metrics_table(metrics)
        assert "Model" in result
        assert "F1" in result

    def test_empty_metrics(self):
        from recast.helpers import format_metrics_table
        result = format_metrics_table({})
        assert isinstance(result, str)

    def test_multiple_models_sorted(self):
        from recast.helpers import format_metrics_table
        metrics = {
            "z_model": {
                "node_precision": {"mean": 0.8, "std": 0.1},
                "node_recall": {"mean": 0.7, "std": 0.15},
                "edge_precision": {"mean": 0.6, "std": 0.2},
                "edge_recall": {"mean": 0.5, "std": 0.25},
                "f1": {"mean": 0.55, "std": 0.18},
                "shd": {"mean": 5.0, "std": 2.0},
                "normalized_shd": {"mean": 0.1, "std": 0.05},
                "sample_count": 100,
            },
            "a_model": {
                "node_precision": {"mean": 0.9, "std": 0.1},
                "node_recall": {"mean": 0.8, "std": 0.15},
                "edge_precision": {"mean": 0.7, "std": 0.2},
                "edge_recall": {"mean": 0.6, "std": 0.25},
                "f1": {"mean": 0.65, "std": 0.18},
                "shd": {"mean": 3.0, "std": 1.0},
                "normalized_shd": {"mean": 0.05, "std": 0.03},
                "sample_count": 100,
            },
        }
        result = format_metrics_table(metrics)
        # a_model should appear before z_model (sorted)
        assert result.index("a_model") < result.index("z_model")


class TestGraphMetricsEdgeCases:
    """Test edge cases for graph metrics."""

    def test_empty_ground_truth_empty_pred_raises(self):
        """Both empty raises due to CDT limitation with empty matrices."""
        # CDT's precision_recall doesn't handle empty graphs
        with pytest.raises(ValueError):
            calculate_graph_metrics([], [])

    def test_empty_ground_truth_nonempty_pred(self):
        """Empty ground truth with predictions - CDT handles this with warning."""
        # CDT issues a warning but computes metrics
        p, r, f1, shd, nshd = calculate_graph_metrics(
            [],
            [{"source": "A", "target": "B"}]
        )
        assert shd >= 0

    def test_nonempty_ground_truth_empty_pred(self):
        """Non-empty ground truth with no predictions should have SHD > 0."""
        p, r, f1, shd, nshd = calculate_graph_metrics(
            [{"source": "A", "target": "B"}],
            []
        )
        assert shd > 0

    def test_single_edge_match(self):
        """Single edge that matches perfectly."""
        edges = [{"source": "X", "target": "Y"}]
        p, r, f1, shd, nshd = calculate_graph_metrics(edges, edges)
        assert shd == 0.0

    def test_reversed_edge(self):
        """Reversed edge direction should be wrong."""
        gt = [{"source": "A", "target": "B"}]
        pred = [{"source": "B", "target": "A"}]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert shd > 0  # Should be penalized

    def test_self_loop_handling(self):
        """Self-loops should be handled gracefully."""
        gt = [{"source": "A", "target": "A"}]  # self-loop
        pred = [{"source": "A", "target": "A"}]
        # Should not crash
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert isinstance(shd, float)

    def test_large_graph(self):
        """Test with a larger graph."""
        import string
        nodes = list(string.ascii_uppercase)[:10]
        gt = [{"source": nodes[i], "target": nodes[i+1]} for i in range(9)]
        pred = [{"source": nodes[i], "target": nodes[i+1]} for i in range(9)]
        p, r, f1, shd, nshd = calculate_graph_metrics(gt, pred)
        assert shd == 0.0


class TestCalculateFullMetricsEdgeCases:
    """Test edge cases for full metrics calculation."""

    def test_empty_graphs_raises(self):
        """Empty graphs raise due to CDT limitation."""
        with pytest.raises(ValueError):
            calculate_full_metrics([], [])

    def test_completely_disjoint_nodes(self):
        gt = [{"source": "A", "target": "B"}]
        pred = [{"source": "X", "target": "Y"}]
        result = calculate_full_metrics(gt, pred)
        assert result["node_precision"] == 0.0
        assert result["node_recall"] == 0.0

    def test_subset_of_nodes(self):
        """Prediction has subset of ground truth nodes."""
        gt = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        pred = [{"source": "A", "target": "B"}]
        result = calculate_full_metrics(gt, pred)
        # pred nodes: A, B (2) - both in gt
        # gt nodes: A, B, C (3)
        assert result["node_precision"] == 1.0  # 2/2
        assert abs(result["node_recall"] - 2/3) < 0.01  # 2/3

    def test_superset_of_nodes(self):
        """Prediction has superset of ground truth nodes."""
        gt = [{"source": "A", "target": "B"}]
        pred = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        result = calculate_full_metrics(gt, pred)
        # pred nodes: A, B, C (3) - 2 in gt
        # gt nodes: A, B (2) - both found
        assert abs(result["node_precision"] - 2/3) < 0.01  # 2/3
        assert result["node_recall"] == 1.0  # 2/2
