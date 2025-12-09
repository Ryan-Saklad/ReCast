"""Tests for the ReCast benchmark module."""

import json

import pytest

from recast.benchmark import (
    prepare_causal_graph_prompt,
    process_answer,
    validate_graph_response,
)


class TestPrepareCausalGraphPrompt:
    """Test prompt preparation."""

    def test_without_node_names(self):
        prompt = prepare_causal_graph_prompt({"A", "B", "C"}, provide_node_names=False)
        assert isinstance(prompt, str)
        assert "3" in prompt  # NUM_NODES replaced

    def test_with_node_names(self):
        prompt = prepare_causal_graph_prompt({"A", "B"}, provide_node_names=True)
        assert isinstance(prompt, str)
        # Should contain JSON with node names
        assert "A" in prompt or "B" in prompt

    def test_node_count_replaced(self):
        nodes = {"X", "Y", "Z", "W"}
        prompt = prepare_causal_graph_prompt(nodes, provide_node_names=False)
        assert "4" in prompt


class TestProcessAnswer:
    """Test answer processing/cleanup."""

    def test_none_returns_none(self):
        assert process_answer(None) is None

    def test_strips_whitespace(self):
        result = process_answer("  {\"test\": 1}  ")
        assert result == '{"test": 1}'

    def test_removes_boxed_wrapper(self):
        result = process_answer('\\boxed{{"relationships": []}}')
        assert result == '{"relationships": []}'

    def test_removes_json_markdown(self):
        result = process_answer('```json\n{"relationships": []}\n```')
        assert result == '\n{"relationships": []}\n'

    def test_preserves_valid_json(self):
        original = '{"relationships": [{"source": "A", "sink": "B"}]}'
        result = process_answer(original)
        assert result == original


class TestValidateGraphResponse:
    """Test response validation."""

    def test_valid_response_passes(self):
        answer = '{"relationships": [{"source": "A", "sink": "B"}]}'
        assert validate_graph_response(answer) is True

    def test_empty_relationships_valid(self):
        answer = '{"relationships": []}'
        assert validate_graph_response(answer) is True

    def test_none_returns_false(self):
        assert validate_graph_response(None) is False

    def test_invalid_json_returns_false(self):
        assert validate_graph_response("not json") is False

    def test_missing_relationships_key_returns_false(self):
        answer = '{"edges": []}'
        assert validate_graph_response(answer) is False

    def test_relationships_not_list_returns_false(self):
        answer = '{"relationships": "invalid"}'
        assert validate_graph_response(answer) is False

    def test_missing_source_returns_false(self):
        answer = '{"relationships": [{"sink": "B"}]}'
        assert validate_graph_response(answer) is False

    def test_missing_sink_returns_false(self):
        answer = '{"relationships": [{"source": "A"}]}'
        assert validate_graph_response(answer) is False

    def test_with_node_names_requires_int_indices(self):
        # When provide_node_names=True, source/sink should be integers
        answer = '{"relationships": [{"source": 1, "sink": 2}]}'
        assert validate_graph_response(answer, provide_node_names=True) is True

    def test_with_node_names_rejects_string_indices(self):
        answer = '{"relationships": [{"source": "A", "sink": "B"}]}'
        assert validate_graph_response(answer, provide_node_names=True) is False

    def test_with_node_names_validates_range(self):
        answer = '{"relationships": [{"source": 1, "sink": 5}]}'
        # With 3 nodes, index 5 is out of range
        assert validate_graph_response(answer, provide_node_names=True, num_nodes=3) is False

    def test_with_node_names_valid_range(self):
        answer = '{"relationships": [{"source": 1, "sink": 3}]}'
        assert validate_graph_response(answer, provide_node_names=True, num_nodes=3) is True

    def test_handles_boxed_wrapper(self):
        answer = '\\boxed{{"relationships": []}}'
        assert validate_graph_response(answer) is True

    def test_handles_json_markdown(self):
        answer = '```json\n{"relationships": []}\n```'
        assert validate_graph_response(answer) is True

    def test_multiple_relationships(self):
        answer = json.dumps({
            "relationships": [
                {"source": "A", "sink": "B"},
                {"source": "B", "sink": "C"},
                {"source": "A", "sink": "C"},
            ]
        })
        assert validate_graph_response(answer) is True


class TestValidationEdgeCases:
    """Edge cases for validation."""

    def test_relationship_with_extra_fields_valid(self):
        """Extra fields should be allowed."""
        answer = '{"relationships": [{"source": "A", "sink": "B", "label": "causes"}]}'
        assert validate_graph_response(answer) is True

    def test_empty_string_returns_false(self):
        assert validate_graph_response("") is False

    def test_whitespace_only_returns_false(self):
        assert validate_graph_response("   ") is False

    def test_nested_json_structure(self):
        """Response with extra top-level keys should be valid."""
        answer = json.dumps({
            "relationships": [{"source": "A", "sink": "B"}],
            "metadata": {"confidence": 0.9}
        })
        assert validate_graph_response(answer) is True
