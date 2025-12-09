"""Tests for the ReCast evaluator module."""

import json

import pytest
import yaml

from recast.evaluator import (
    verify_aggregate_output_format,
    parse_aggregate_scores,
    parse_fine_grained_yaml,
    _format_graph_for_prompt,
)


class TestVerifyAggregateOutputFormat:
    """Test LLM judge output format verification."""

    def test_valid_format_passes(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 4,
                "causal_recall": 3,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is True

    def test_all_scores_at_minimum(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 1,
                "causal_recall": 1,
                "semantic_similarity": 1
            }
        })
        assert verify_aggregate_output_format(output) is True

    def test_all_scores_at_maximum(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 5,
                "causal_recall": 5,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is True

    def test_missing_scores_key_fails(self):
        output = json.dumps({
            "causal_precision": 4,
            "causal_recall": 3,
            "semantic_similarity": 5
        })
        assert verify_aggregate_output_format(output) is False

    def test_missing_causal_precision_fails(self):
        output = json.dumps({
            "scores": {
                "causal_recall": 3,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_missing_causal_recall_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 4,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_missing_semantic_similarity_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 4,
                "causal_recall": 3
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_extra_score_key_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 4,
                "causal_recall": 3,
                "semantic_similarity": 5,
                "extra_score": 2
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_score_below_range_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 0,
                "causal_recall": 3,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_score_above_range_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 4,
                "causal_recall": 6,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_float_score_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": 4.5,
                "causal_recall": 3,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_string_score_fails(self):
        output = json.dumps({
            "scores": {
                "causal_precision": "4",
                "causal_recall": 3,
                "semantic_similarity": 5
            }
        })
        assert verify_aggregate_output_format(output) is False

    def test_invalid_json_fails(self):
        assert verify_aggregate_output_format("not json") is False

    def test_empty_string_fails(self):
        assert verify_aggregate_output_format("") is False

    def test_handles_json_markdown_wrapper(self):
        output = '```json\n{"scores": {"causal_precision": 4, "causal_recall": 3, "semantic_similarity": 5}}\n```'
        assert verify_aggregate_output_format(output) is True

    def test_handles_whitespace(self):
        output = '  {"scores": {"causal_precision": 4, "causal_recall": 3, "semantic_similarity": 5}}  '
        assert verify_aggregate_output_format(output) is True

    def test_null_scores_fails(self):
        output = json.dumps({"scores": None})
        assert verify_aggregate_output_format(output) is False

    def test_scores_as_list_fails(self):
        output = json.dumps({"scores": [4, 3, 5]})
        assert verify_aggregate_output_format(output) is False


class TestParseAggregateScores:
    """Test parsing evaluation scores from LLM output."""

    def test_parse_valid_scores(self):
        answer = json.dumps({
            "scores": {
                "causal_precision": 4,
                "causal_recall": 3,
                "semantic_similarity": 5
            }
        })
        scores = parse_aggregate_scores(answer)
        assert scores["causal_precision"] == 4
        assert scores["causal_recall"] == 3
        assert scores["semantic_similarity"] == 5

    def test_parse_with_markdown_wrapper(self):
        answer = '```json\n{"scores": {"causal_precision": 2, "causal_recall": 2, "semantic_similarity": 3}}\n```'
        scores = parse_aggregate_scores(answer)
        assert scores["causal_precision"] == 2
        assert scores["causal_recall"] == 2
        assert scores["semantic_similarity"] == 3

    def test_parse_minimum_scores(self):
        answer = json.dumps({
            "scores": {
                "causal_precision": 1,
                "causal_recall": 1,
                "semantic_similarity": 1
            }
        })
        scores = parse_aggregate_scores(answer)
        assert all(v == 1 for v in scores.values())

    def test_parse_maximum_scores(self):
        answer = json.dumps({
            "scores": {
                "causal_precision": 5,
                "causal_recall": 5,
                "semantic_similarity": 5
            }
        })
        scores = parse_aggregate_scores(answer)
        assert all(v == 5 for v in scores.values())

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_aggregate_scores("not json")

    def test_missing_scores_key_raises(self):
        answer = json.dumps({"wrong_key": {}})
        with pytest.raises((KeyError, TypeError)):
            parse_aggregate_scores(answer)

    def test_returns_dict_type(self):
        answer = json.dumps({
            "scores": {
                "causal_precision": 3,
                "causal_recall": 4,
                "semantic_similarity": 3
            }
        })
        scores = parse_aggregate_scores(answer)
        assert isinstance(scores, dict)
        assert len(scores) == 3


class TestParseFineGrainedYaml:
    """Test fine-grained YAML output parsing."""

    def test_parse_node_precision(self):
        yaml_str = """```yaml
node_precision_evaluations:
  - node_number: 1
    graph_evaluation:
      presence_label: PRESENCE_STRONG_MATCH
      semantic_label: SEMANTIC_STRONG
      abstraction_label: ABSTRACTION_ALIGNED
    text_evaluation:
      presence_label: PRESENCE_STRONG_MATCH
      semantic_label: SEMANTIC_STRONG
      abstraction_label: ABSTRACTION_ALIGNED
```"""
        result = parse_fine_grained_yaml(yaml_str)
        assert result is not None
        assert "node_precision_evaluations" in result
        assert len(result["node_precision_evaluations"]) == 1
        assert result["node_precision_evaluations"][0]["node_number"] == 1

    def test_parse_node_recall(self):
        yaml_str = """```yaml
node_recall_evaluations:
  - node_number: 1
    importance_label: IMPORTANCE_CORE
    presence_label: PRESENCE_STRONG_MATCH
    semantic_label: SEMANTIC_COMPLETE
    abstraction_label: ABSTRACTION_ALIGNED
```"""
        result = parse_fine_grained_yaml(yaml_str)
        assert result is not None
        assert "node_recall_evaluations" in result
        assert result["node_recall_evaluations"][0]["importance_label"] == "IMPORTANCE_CORE"

    def test_parse_edge_precision(self):
        yaml_str = """```yaml
edge_precision_evaluations:
  - edge_number: 1
    graph_evaluation:
      presence_label: PRESENCE_STRONG_MATCH
      directionality_label: DIRECTION_CORRECT
      abstraction_label: ABSTRACTION_ALIGNED
    text_evaluation:
      presence_label: PRESENCE_GRAPH_ONLY
      inference_label: INFERENCE_DIRECT
      abstraction_label: ABSTRACTION_ALIGNED
```"""
        result = parse_fine_grained_yaml(yaml_str)
        assert result is not None
        assert "edge_precision_evaluations" in result
        eval_item = result["edge_precision_evaluations"][0]
        assert eval_item["graph_evaluation"]["directionality_label"] == "DIRECTION_CORRECT"
        assert eval_item["text_evaluation"]["inference_label"] == "INFERENCE_DIRECT"

    def test_parse_edge_recall(self):
        yaml_str = """```yaml
edge_recall_evaluations:
  - edge_number: 1
    importance_label: IMPORTANCE_CENTRAL
    presence_label: PRESENCE_WEAK_MATCH
    directionality_label: DIRECTION_CORRECT
    abstraction_label: ABSTRACTION_BROADER
```"""
        result = parse_fine_grained_yaml(yaml_str)
        assert result is not None
        assert "edge_recall_evaluations" in result
        assert result["edge_recall_evaluations"][0]["directionality_label"] == "DIRECTION_CORRECT"

    def test_parse_without_yaml_wrapper(self):
        yaml_str = """node_precision_evaluations:
  - node_number: 1
    graph_evaluation:
      presence_label: PRESENCE_NO_MATCH
      semantic_label: SEMANTIC_NA
      abstraction_label: ABSTRACTION_NA
    text_evaluation:
      presence_label: PRESENCE_NO_MATCH
      semantic_label: SEMANTIC_NA
      abstraction_label: ABSTRACTION_NA"""
        result = parse_fine_grained_yaml(yaml_str)
        assert result is not None
        assert "node_precision_evaluations" in result

    def test_parse_multiple_items(self):
        yaml_str = """```yaml
node_precision_evaluations:
  - node_number: 1
    graph_evaluation:
      presence_label: PRESENCE_STRONG_MATCH
      semantic_label: SEMANTIC_STRONG
      abstraction_label: ABSTRACTION_ALIGNED
    text_evaluation:
      presence_label: PRESENCE_STRONG_MATCH
      semantic_label: SEMANTIC_STRONG
      abstraction_label: ABSTRACTION_ALIGNED
  - node_number: 2
    graph_evaluation:
      presence_label: PRESENCE_WEAK_MATCH
      semantic_label: SEMANTIC_MODERATE
      abstraction_label: ABSTRACTION_BROADER
    text_evaluation:
      presence_label: PRESENCE_NO_MATCH
      semantic_label: SEMANTIC_NA
      abstraction_label: ABSTRACTION_NA
```"""
        result = parse_fine_grained_yaml(yaml_str)
        assert result is not None
        assert len(result["node_precision_evaluations"]) == 2

    def test_invalid_yaml_returns_none(self):
        result = parse_fine_grained_yaml("not: valid: yaml: [[[")
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_fine_grained_yaml("")
        assert result is None


class TestFormatGraphForPrompt:
    """Test graph formatting for evaluation prompts."""

    def test_simple_graph(self):
        graph_json = '[{"source": "A", "sink": "B"}, {"source": "B", "sink": "C"}]'
        result = _format_graph_for_prompt(graph_json)
        assert "Nodes:" in result
        assert "Edges:" in result
        assert "1. A" in result
        assert "2. B" in result
        assert "3. C" in result
        assert "1. A -> B" in result
        assert "2. B -> C" in result

    def test_handles_target_key(self):
        graph_json = '[{"source": "X", "target": "Y"}]'
        result = _format_graph_for_prompt(graph_json)
        assert "X" in result
        assert "Y" in result
        assert "X -> Y" in result

    def test_handles_dict_with_relationships(self):
        graph_json = '{"relationships": [{"source": "A", "sink": "B"}]}'
        result = _format_graph_for_prompt(graph_json)
        assert "A" in result
        assert "B" in result

    def test_nodes_sorted_alphabetically(self):
        graph_json = '[{"source": "Z", "sink": "A"}, {"source": "M", "sink": "B"}]'
        result = _format_graph_for_prompt(graph_json)
        lines = result.split("\n")
        node_lines = [l for l in lines if l.startswith("1.") or l.startswith("2.") or l.startswith("3.") or l.startswith("4.")]
        # First four should be nodes A, B, M, Z in that order
        assert "1. A" in result
        assert "2. B" in result
        assert "3. M" in result
        assert "4. Z" in result

    def test_invalid_json_returns_original(self):
        invalid = "not valid json"
        result = _format_graph_for_prompt(invalid)
        assert result == invalid

    def test_empty_graph(self):
        graph_json = '[]'
        result = _format_graph_for_prompt(graph_json)
        assert "Nodes:" in result
        assert "Edges:" in result
