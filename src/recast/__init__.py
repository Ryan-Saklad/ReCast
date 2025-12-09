"""ReCast: Real-world Causal Structure Inference Benchmark."""

__version__ = "0.1.0"

from recast.dataset import (
    load_dataset,
    load_responses,
    load_evaluations,
    load_responses_from_db,
    load_evaluations_from_db,
    get_paper_results,
    export_to_parquet,
    ReCastSample,
    ReCastDataset,
    ModelResponse,
    Evaluation,
)
from recast.helpers import (
    calculate_graph_metrics,
    calculate_full_metrics,
    generate_random_baseline,
    load_prompt,
    create_db,
    nodes_to_node_dict,
    get_model_metrics,
    get_full_model_metrics,
    format_metrics_table,
)
from recast.benchmark import (
    run_benchmark,
    run_benchmark_for_sample,
    generate_causal_graph,
    correct_causal_graph,
    validate_graph_response,
    process_answer,
)
from recast.evaluator import (
    run_aggregate_evaluation,
    run_fine_grained_evaluation,
    evaluate_all_responses,
    get_llm_judge_response,
    parse_aggregate_scores,
    parse_fine_grained_yaml,
    verify_aggregate_output_format,
)

__all__ = [
    # Dataset
    "load_dataset",
    "load_responses",
    "load_evaluations",
    "load_responses_from_db",
    "load_evaluations_from_db",
    "get_paper_results",
    "export_to_parquet",
    "ReCastSample",
    "ReCastDataset",
    "ModelResponse",
    "Evaluation",
    # Metrics & Helpers
    "calculate_graph_metrics",
    "calculate_full_metrics",
    "generate_random_baseline",
    "load_prompt",
    "create_db",
    "nodes_to_node_dict",
    "get_model_metrics",
    "get_full_model_metrics",
    "format_metrics_table",
    # Benchmark
    "run_benchmark",
    "run_benchmark_for_sample",
    "generate_causal_graph",
    "correct_causal_graph",
    "validate_graph_response",
    "process_answer",
    # Evaluation
    "run_aggregate_evaluation",
    "run_fine_grained_evaluation",
    "evaluate_all_responses",
    "get_llm_judge_response",
    "parse_aggregate_scores",
    "parse_fine_grained_yaml",
    "verify_aggregate_output_format",
]
