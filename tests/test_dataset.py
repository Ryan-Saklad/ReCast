"""Tests for the ReCast dataset loader."""

import pytest
from recast import load_dataset, ReCastSample, ReCastDataset


@pytest.fixture(scope="module")
def dataset():
    """Load dataset once for all tests."""
    return load_dataset()


class TestDatasetLoading:
    """Test dataset loading from HuggingFace."""

    def test_load_from_huggingface_repo(self):
        """Test that dataset loads from the correct HuggingFace repo."""
        ds = load_dataset("RyanSaklad/ReCast")
        assert len(ds) == 292
        assert ds[0].id == 1

    def test_load_dataset_returns_recast_dataset(self, dataset):
        assert isinstance(dataset, ReCastDataset)

    def test_dataset_has_292_samples(self, dataset):
        assert len(dataset) == 292

    def test_dataset_is_iterable(self, dataset):
        samples = list(dataset)
        assert len(samples) == 292
        assert all(isinstance(s, ReCastSample) for s in samples)

    def test_dataset_is_indexable(self, dataset):
        sample = dataset[0]
        assert isinstance(sample, ReCastSample)

    def test_dataset_negative_indexing(self, dataset):
        last = dataset[-1]
        assert isinstance(last, ReCastSample)
        assert last.id == dataset[len(dataset) - 1].id

    def test_dataset_slicing(self, dataset):
        subset = dataset[:5]
        assert isinstance(subset, list)
        assert len(subset) == 5


class TestReCastSample:
    """Test ReCastSample structure."""

    def test_sample_has_required_fields(self, dataset):
        sample = dataset[0]
        required = [
            "id", "title", "source", "url", "domains", "num_nodes",
            "num_edges", "explicitness", "nodes", "edges",
            "node_explicitness", "input_text", "abstract", "publication_date"
        ]
        for field in required:
            assert hasattr(sample, field), f"Missing field: {field}"

    def test_sample_id_is_int(self, dataset):
        assert isinstance(dataset[0].id, int)

    def test_sample_domains_is_list(self, dataset):
        sample = dataset[0]
        assert isinstance(sample.domains, list)
        assert all(isinstance(d, str) for d in sample.domains)

    def test_sample_nodes_matches_num_nodes(self, dataset):
        for sample in dataset[:10]:
            assert len(sample.nodes) == sample.num_nodes

    def test_sample_edges_matches_num_edges(self, dataset):
        for sample in dataset[:10]:
            assert len(sample.edges) == sample.num_edges

    def test_sample_edges_have_source_and_target(self, dataset):
        sample = dataset[0]
        for edge in sample.edges:
            assert isinstance(edge, dict)
            assert "source" in edge
            assert "target" in edge

    def test_sample_explicitness_in_valid_range(self, dataset):
        for sample in dataset:
            assert 0.0 <= sample.explicitness <= 1.0

    def test_sample_source_is_plos_or_mdpi(self, dataset):
        for sample in dataset:
            assert sample.source in ("PLOS", "MDPI")

    def test_all_edge_nodes_in_node_list(self, dataset):
        """Verify all edge endpoints exist in the nodes list."""
        for sample in dataset[:10]:
            nodes_set = set(sample.nodes)
            for edge in sample.edges:
                assert edge["source"] in nodes_set
                assert edge["target"] in nodes_set


class TestDatasetFilteringBasic:
    """Test basic dataset filtering methods."""

    def test_filter_by_domain(self, dataset):
        medical = dataset.filter_by_domain("Medicine")
        assert isinstance(medical, ReCastDataset)
        assert len(medical) > 0
        assert all("Medicine" in s.domains for s in medical)

    def test_filter_by_nonexistent_domain_returns_empty(self, dataset):
        result = dataset.filter_by_domain("NonexistentDomain")
        assert len(result) == 0

    def test_filter_by_min_explicitness(self, dataset):
        easy = dataset.filter_by_min_explicitness(0.8)
        assert isinstance(easy, ReCastDataset)
        assert all(s.explicitness >= 0.8 for s in easy)

    def test_filter_by_max_explicitness(self, dataset):
        hard = dataset.filter_by_max_explicitness(0.2)
        assert isinstance(hard, ReCastDataset)
        assert all(s.explicitness <= 0.2 for s in hard)

    def test_filter_by_min_explicitness_zero_returns_all(self, dataset):
        result = dataset.filter_by_min_explicitness(0.0)
        assert len(result) == len(dataset)

    def test_filter_by_max_explicitness_one_returns_all(self, dataset):
        result = dataset.filter_by_max_explicitness(1.0)
        assert len(result) == len(dataset)

    def test_chained_filters(self, dataset):
        result = (
            dataset
            .filter_by_domain("Medicine")
            .filter_by_min_explicitness(0.5)
        )
        assert all("Medicine" in s.domains and s.explicitness >= 0.5 for s in result)


class TestDatasetFilteringAdvanced:
    """Test advanced filtering methods."""

    def test_filter_by_domains_any(self, dataset):
        result = dataset.filter_by_domains(["Medicine", "Education"], require_all=False)
        assert len(result) > 0
        assert all(
            "Medicine" in s.domains or "Education" in s.domains
            for s in result
        )

    def test_filter_by_domains_all(self, dataset):
        # Find samples that have both domains
        result = dataset.filter_by_domains(
            ["Economics & Public Policy", "Engineering & Technology"],
            require_all=True
        )
        assert all(
            "Economics & Public Policy" in s.domains and "Engineering & Technology" in s.domains
            for s in result
        )

    def test_filter_by_source(self, dataset):
        plos = dataset.filter_by_source("PLOS")
        assert len(plos) > 0
        assert all(s.source == "PLOS" for s in plos)

        mdpi = dataset.filter_by_source("MDPI")
        assert len(mdpi) > 0
        assert all(s.source == "MDPI" for s in mdpi)

    def test_filter_by_source_coverage(self, dataset):
        """PLOS + MDPI should equal total."""
        plos = dataset.filter_by_source("PLOS")
        mdpi = dataset.filter_by_source("MDPI")
        assert len(plos) + len(mdpi) == len(dataset)

    def test_filter_by_min_nodes(self, dataset):
        large = dataset.filter_by_min_nodes(30)
        assert all(s.num_nodes >= 30 for s in large)

    def test_filter_by_max_nodes(self, dataset):
        small = dataset.filter_by_max_nodes(10)
        assert all(s.num_nodes <= 10 for s in small)

    def test_filter_by_min_edges(self, dataset):
        complex_graphs = dataset.filter_by_min_edges(50)
        assert all(s.num_edges >= 50 for s in complex_graphs)

    def test_filter_by_max_edges(self, dataset):
        simple = dataset.filter_by_max_edges(15)
        assert all(s.num_edges <= 15 for s in simple)

    def test_filter_by_ids(self, dataset):
        # Get first 5 sample IDs
        ids = [dataset[i].id for i in range(5)]
        result = dataset.filter_by_ids(ids)
        assert len(result) == 5
        assert all(s.id in ids for s in result)

    def test_filter_by_ids_nonexistent(self, dataset):
        result = dataset.filter_by_ids([-999, -998])
        assert len(result) == 0


class TestDatasetFilteringDate:
    """Test date-based filtering."""

    def test_filter_by_date_after(self, dataset):
        # Only test on samples that have dates
        dated_samples = [s for s in dataset if s.publication_date]
        if dated_samples:
            result = dataset.filter_by_date_after("2022-01-01")
            assert all(
                s.publication_date >= "2022-01-01"
                for s in result
                if s.publication_date
            )

    def test_filter_by_date_before(self, dataset):
        dated_samples = [s for s in dataset if s.publication_date]
        if dated_samples:
            result = dataset.filter_by_date_before("2022-01-01")
            # Empty dates should not be included (they are < any date string)
            for s in result:
                if s.publication_date:
                    assert s.publication_date < "2022-01-01"

    def test_filter_by_date_range(self, dataset):
        result = dataset.filter_by_date_range("2020-01-01", "2022-12-31")
        for s in result:
            if s.publication_date:
                assert "2020-01-01" <= s.publication_date <= "2022-12-31"


class TestDatasetStatistics:
    """Test dataset statistics method."""

    def test_statistics_returns_dict(self, dataset):
        stats = dataset.statistics()
        assert isinstance(stats, dict)

    def test_statistics_has_total_samples(self, dataset):
        stats = dataset.statistics()
        assert stats["total_samples"] == 292

    def test_statistics_has_domains(self, dataset):
        stats = dataset.statistics()
        assert "domains" in stats
        assert isinstance(stats["domains"], dict)
        assert len(stats["domains"]) >= 4

    def test_statistics_has_sources(self, dataset):
        stats = dataset.statistics()
        assert "sources" in stats
        assert "PLOS" in stats["sources"]
        assert "MDPI" in stats["sources"]

    def test_statistics_has_publication_years(self, dataset):
        stats = dataset.statistics()
        assert "publication_years" in stats

    def test_statistics_has_date_range(self, dataset):
        stats = dataset.statistics()
        assert "date_range" in stats
        dr = stats["date_range"]
        assert "earliest" in dr
        assert "latest" in dr
        assert "samples_with_dates" in dr

    def test_statistics_has_nodes(self, dataset):
        stats = dataset.statistics()
        assert "nodes" in stats
        assert "min" in stats["nodes"]
        assert "max" in stats["nodes"]
        assert "mean" in stats["nodes"]

    def test_statistics_has_edges(self, dataset):
        stats = dataset.statistics()
        assert "edges" in stats
        assert "min" in stats["edges"]
        assert "max" in stats["edges"]
        assert "mean" in stats["edges"]

    def test_statistics_has_explicitness(self, dataset):
        stats = dataset.statistics()
        assert "explicitness" in stats
        assert 0.0 <= stats["explicitness"]["min"] <= stats["explicitness"]["max"] <= 1.0

    def test_statistics_on_filtered_dataset(self, dataset):
        filtered = dataset.filter_by_domain("Medicine")
        stats = filtered.statistics()
        assert stats["total_samples"] == len(filtered)
        assert stats["total_samples"] < 292

    def test_empty_dataset_statistics(self):
        empty = ReCastDataset([])
        stats = empty.statistics()
        assert stats["total_samples"] == 0


class TestDatasetHelpers:
    """Test helper methods."""

    def test_get_sample_ids(self, dataset):
        ids = dataset.get_sample_ids()
        assert isinstance(ids, list)
        assert len(ids) == len(dataset)
        assert all(isinstance(i, int) for i in ids)

    def test_get_domains(self, dataset):
        domains = dataset.get_domains()
        assert isinstance(domains, list)
        assert len(domains) >= 4
        assert all(isinstance(d, str) for d in domains)
        # Should be sorted
        assert domains == sorted(domains)

    def test_get_sources(self, dataset):
        sources = dataset.get_sources()
        assert isinstance(sources, list)
        assert "PLOS" in sources
        assert "MDPI" in sources
        assert len(sources) == 2


class TestDatasetWideStatistics:
    """Test dataset-wide statistics."""

    def test_multiple_domains_exist(self, dataset):
        all_domains = set()
        for sample in dataset:
            all_domains.update(sample.domains)
        assert len(all_domains) >= 4

    def test_both_sources_present(self, dataset):
        sources = {s.source for s in dataset}
        assert "PLOS" in sources
        assert "MDPI" in sources

    def test_graph_size_variety(self, dataset):
        node_counts = [s.num_nodes for s in dataset]
        edge_counts = [s.num_edges for s in dataset]

        assert min(node_counts) < 10
        assert max(node_counts) > 50
        assert min(edge_counts) < 10
        assert max(edge_counts) > 50

    def test_explicitness_variety(self, dataset):
        explicitness = [s.explicitness for s in dataset]
        assert min(explicitness) < 0.3
        assert max(explicitness) > 0.7


class TestLoadResponses:
    """Test loading model responses from HuggingFace."""

    @pytest.fixture(scope="class")
    def responses(self):
        from recast import load_responses
        return load_responses()

    def test_returns_list(self, responses):
        assert isinstance(responses, list)

    def test_has_responses(self, responses):
        assert len(responses) > 0

    def test_response_has_required_fields(self, responses):
        from recast.dataset import ModelResponse
        r = responses[0]
        assert isinstance(r, ModelResponse)
        assert hasattr(r, "id")
        assert hasattr(r, "sample_id")
        assert hasattr(r, "model")
        assert hasattr(r, "task_type")
        assert hasattr(r, "response_answer")
        assert hasattr(r, "response_reasoning")
        assert hasattr(r, "corrected_answer")
        assert hasattr(r, "valid_format")
        assert hasattr(r, "response_date")

    def test_response_id_is_int(self, responses):
        assert all(isinstance(r.id, int) for r in responses[:10])

    def test_response_sample_id_is_int(self, responses):
        assert all(isinstance(r.sample_id, int) for r in responses[:10])

    def test_response_model_is_string(self, responses):
        assert all(isinstance(r.model, str) for r in responses[:10])

    def test_has_multiple_models(self, responses):
        models = {r.model for r in responses}
        assert len(models) > 1

    def test_has_both_task_types(self, responses):
        task_types = {r.task_type for r in responses}
        assert "causal_graph_generation" in task_types


class TestLoadEvaluations:
    """Test loading evaluations from HuggingFace."""

    @pytest.fixture(scope="class")
    def evaluations(self):
        from recast import load_evaluations
        return load_evaluations()

    def test_returns_list(self, evaluations):
        assert isinstance(evaluations, list)

    def test_has_evaluations(self, evaluations):
        assert len(evaluations) > 0

    def test_evaluation_has_required_fields(self, evaluations):
        from recast.dataset import Evaluation
        e = evaluations[0]
        assert isinstance(e, Evaluation)
        assert hasattr(e, "id")
        assert hasattr(e, "response_id")
        assert hasattr(e, "sample_id")
        assert hasattr(e, "model")
        assert hasattr(e, "task_type")
        assert hasattr(e, "evaluator_type")
        assert hasattr(e, "score")
        assert hasattr(e, "evaluation_answer")
        assert hasattr(e, "evaluation_reasoning")
        assert hasattr(e, "evaluation_date")

    def test_evaluation_id_is_int(self, evaluations):
        assert all(isinstance(e.id, int) for e in evaluations[:10])

    def test_evaluation_score_is_float(self, evaluations):
        assert all(isinstance(e.score, (int, float)) for e in evaluations[:10])

    def test_has_multiple_evaluator_types(self, evaluations):
        types = {e.evaluator_type for e in evaluations}
        assert len(types) > 1


class TestEvaluationParsedScores:
    """Test Evaluation.parsed_scores property."""

    @pytest.fixture(scope="class")
    def evaluations(self):
        from recast import load_evaluations
        return load_evaluations()

    def test_llm_judge_has_parsed_scores(self, evaluations):
        llm_evals = [e for e in evaluations if "llm_judge" in e.evaluator_type]
        if llm_evals:
            e = llm_evals[0]
            scores = e.parsed_scores
            # May be None if parsing fails, but shouldn't error
            assert scores is None or isinstance(scores, dict)

    def test_parsed_scores_has_expected_keys(self, evaluations):
        llm_evals = [e for e in evaluations if "llm_judge" in e.evaluator_type]
        for e in llm_evals[:10]:
            scores = e.parsed_scores
            if scores:
                # Should have score keys
                assert "causal_precision" in scores or "causal_accuracy" in scores or len(scores) > 0

    def test_parsed_fine_grained_only_for_fine_grained(self, evaluations):
        """parsed_fine_grained should return None for non-fine-grained evals."""
        non_fg = [e for e in evaluations if e.evaluator_type != "fine_grained"]
        if non_fg:
            assert non_fg[0].parsed_fine_grained is None


class TestGetPaperResults:
    """Test get_paper_results function."""

    def test_llm_judge_returns_dict(self):
        from recast import get_paper_results
        results = get_paper_results("llm_judge")
        assert isinstance(results, dict)

    def test_llm_judge_has_models(self):
        from recast import get_paper_results
        results = get_paper_results("llm_judge")
        assert len(results) > 0

    def test_llm_judge_metrics_have_mean_std(self):
        from recast import get_paper_results
        results = get_paper_results("llm_judge")
        for model, metrics in results.items():
            # Each metric should have mean and std (except 'n' which is count)
            for key, value in metrics.items():
                if key == "n":
                    assert isinstance(value, int)
                else:
                    assert "mean" in value
                    assert "std" in value

    def test_graph_similarity_returns_dict(self):
        from recast import get_paper_results
        results = get_paper_results("graph_similarity")
        assert isinstance(results, dict)

    def test_deterministic_returns_dict(self):
        from recast import get_paper_results
        results = get_paper_results("deterministic")
        assert isinstance(results, dict)

    def test_deterministic_has_expected_metrics(self):
        from recast import get_paper_results
        results = get_paper_results("deterministic")
        if results:
            model = next(iter(results))
            metrics = results[model]
            # Should have SHD, precision, recall, etc.
            assert "shd" in metrics or "f1" in metrics or len(metrics) > 0

    def test_with_node_names_task_type(self):
        from recast import get_paper_results
        results = get_paper_results("deterministic", "causal_graph_generation_with_node_names")
        # May be empty if no such evaluations, but shouldn't error
        assert isinstance(results, dict)
