"""LLM-as-judge evaluation for ReCast benchmark responses."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import TypedDict, Literal

import aiosqlite
import yaml

from .helpers import load_prompt, create_db
from .openrouter import call_openrouter

DEFAULT_DB_PATH = "recast_results.db"

EvaluationType = Literal["node_precision", "node_recall", "edge_precision", "edge_recall"]


class EvaluationScores(TypedDict):
    causal_precision: int
    causal_recall: int
    semantic_similarity: int


class FineGrainedNodeEval(TypedDict, total=False):
    node_number: int
    graph_evaluation: dict
    text_evaluation: dict
    importance_label: str
    presence_label: str
    semantic_label: str
    abstraction_label: str


class FineGrainedEdgeEval(TypedDict, total=False):
    edge_number: int
    graph_evaluation: dict
    text_evaluation: dict
    importance_label: str
    presence_label: str
    directionality_label: str
    abstraction_label: str


class FineGrainedResult(TypedDict, total=False):
    node_precision_evaluations: list[FineGrainedNodeEval]
    node_recall_evaluations: list[FineGrainedNodeEval]
    edge_precision_evaluations: list[FineGrainedEdgeEval]
    edge_recall_evaluations: list[FineGrainedEdgeEval]


def verify_aggregate_output_format(output: str) -> bool:
    """Check if aggregate LLM judge output has correct JSON structure."""
    try:
        cleaned = output.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

        if not isinstance(data, dict) or "scores" not in data:
            return False

        scores = data["scores"]
        if not isinstance(scores, dict):
            return False

        required_keys = {"causal_precision", "causal_recall", "semantic_similarity"}
        if set(scores.keys()) != required_keys:
            return False

        for score in scores.values():
            if not isinstance(score, int) or not 1 <= score <= 5:
                return False

        return True
    except json.JSONDecodeError:
        return False


def parse_aggregate_scores(answer: str) -> EvaluationScores:
    """Parse aggregate evaluation scores from LLM judge response."""
    cleaned = answer.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return data["scores"]


def parse_fine_grained_yaml(answer: str) -> FineGrainedResult | None:
    """Parse fine-grained YAML output from LLM judge response."""
    match = re.search(r'```yaml\s*(.*?)\s*```', answer, re.DOTALL)
    if match:
        yaml_str = match.group(1)
    else:
        yaml_str = answer

    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return None


def _format_graph_for_prompt(graph_json: str) -> str:
    """Format graph JSON for evaluation prompts with numbered nodes/edges."""
    try:
        edges = json.loads(graph_json)
        if isinstance(edges, dict):
            edges = edges.get("relationships", [])

        nodes: set[str] = set()
        for edge in edges:
            nodes.add(edge["source"])
            target = edge.get("sink") or edge.get("target")
            if target:
                nodes.add(target)

        sorted_nodes = sorted(nodes)
        node_lines = [f"{i+1}. {node}" for i, node in enumerate(sorted_nodes)]

        edge_lines = []
        for i, edge in enumerate(edges):
            source = edge["source"]
            target = edge.get("sink") or edge.get("target")
            edge_lines.append(f"{i+1}. {source} -> {target}")

        return (
            f"Nodes:\n" + "\n".join(node_lines) + "\n\n"
            f"Edges:\n" + "\n".join(edge_lines)
        )
    except (json.JSONDecodeError, KeyError):
        return graph_json


async def run_aggregate_evaluation(
    response_id: int,
    response_answer: str,
    ground_truth: str,
    input_text: str,
    model: str = "deepseek/deepseek-r1",
    temperature: float = 0.7,
    max_tokens: int = 10000,
    db_path: str = DEFAULT_DB_PATH,
) -> EvaluationScores:
    """Run aggregate LLM-as-judge evaluation on a model response.

    Args:
        response_id: ID of the benchmark response being evaluated
        response_answer: The model's generated causal graph JSON
        ground_truth: The ground truth causal graph JSON
        input_text: The original input text
        model: Evaluator model to use (via OpenRouter)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        db_path: Path to SQLite database

    Returns:
        Dictionary with evaluation scores (causal_precision, causal_recall, semantic_similarity)
    """
    system_prompt = load_prompt("llm_judge").strip()
    user_message = (
        f"# Original Text\n\n{input_text}\n\n"
        f"# Ground Truth Causal Graph\n\n{ground_truth}\n\n"
        f"# Generated Causal Graph\n\n{response_answer}"
    )

    result = await call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        include_reasoning=True,
    )

    if not result.answer:
        raise RuntimeError("No answer returned from LLM")

    if not verify_aggregate_output_format(result.answer):
        raise RuntimeError("Invalid output format returned from LLM")

    # Store evaluation in database
    create_db(db_path)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            INSERT INTO benchmark_evaluations
            (response_id, evaluation_date, evaluator_type, evaluation_answer, evaluation_reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (response_id, datetime.now().isoformat(), f"llm_judge_{model}", result.answer, result.reasoning),
        )
        await conn.commit()

    return parse_aggregate_scores(result.answer)


async def run_fine_grained_evaluation(
    response_id: int,
    response_answer: str,
    ground_truth: str,
    input_text: str,
    evaluation_types: list[EvaluationType] | None = None,
    model: str = "deepseek/deepseek-r1",
    temperature: float = 0.7,
    max_tokens: int = 30000,
    db_path: str = DEFAULT_DB_PATH,
) -> FineGrainedResult:
    """Run fine-grained LLM-as-judge evaluation with separate calls for each metric.

    Makes 4 separate LLM calls (node precision, node recall, edge precision, edge recall)
    to get detailed per-node and per-edge evaluations.

    Args:
        response_id: ID of the benchmark response being evaluated
        response_answer: The model's generated causal graph JSON
        ground_truth: The ground truth causal graph JSON
        input_text: The original input text
        evaluation_types: Which evaluations to run (default: all 4)
        model: Evaluator model to use (via OpenRouter)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate per call
        db_path: Path to SQLite database

    Returns:
        Combined FineGrainedResult with all evaluation types
    """
    if evaluation_types is None:
        evaluation_types = ["node_precision", "node_recall", "edge_precision", "edge_recall"]

    system_prompt = load_prompt("fine_grained_llm_judge").strip()

    # Format graphs with numbered nodes/edges for evaluation
    formatted_gt = _format_graph_for_prompt(ground_truth)
    formatted_response = _format_graph_for_prompt(response_answer)

    combined_result: FineGrainedResult = {}

    async def run_single_evaluation(eval_type: EvaluationType) -> tuple[EvaluationType, dict | None, str | None]:
        eval_type_display = eval_type.replace("_", " ").title()
        user_message = (
            f"# Evaluation Type\n\n{eval_type_display}\n\n"
            f"# Original Text\n\n{input_text}\n\n"
            f"# Ground Truth Causal Graph\n\n{formatted_gt}\n\n"
            f"# Generated Causal Graph\n\n{formatted_response}"
        )

        result = await call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            include_reasoning=True,
        )

        if not result.answer:
            return eval_type, None, result.reasoning

        parsed = parse_fine_grained_yaml(result.answer)
        return eval_type, parsed, result.reasoning

    # Run all evaluation types concurrently
    tasks = [run_single_evaluation(et) for et in evaluation_types]
    results = await asyncio.gather(*tasks)

    # Combine results
    all_reasoning: list[str] = []
    for eval_type, parsed, reasoning in results:
        if parsed:
            key = f"{eval_type}_evaluations"
            if key in parsed:
                combined_result[key] = parsed[key]
        if reasoning:
            all_reasoning.append(f"=== {eval_type} ===\n{reasoning}")

    # Store combined evaluation in database
    create_db(db_path)
    async with aiosqlite.connect(db_path) as conn:
        combined_reasoning = "\n\n".join(all_reasoning) if all_reasoning else None
        await conn.execute(
            """
            INSERT INTO benchmark_evaluations
            (response_id, evaluation_date, evaluator_type, evaluation_answer, evaluation_reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                response_id,
                datetime.now().isoformat(),
                f"fine_grained_{model}",
                yaml.dump(combined_result),
                combined_reasoning,
            ),
        )
        await conn.commit()

    return combined_result


async def evaluate_all_responses(
    evaluation_method: Literal["aggregate", "fine_grained"] = "aggregate",
    model: str = "deepseek/deepseek-r1",
    task_type: str = "causal_graph_generation",
    db_path: str = DEFAULT_DB_PATH,
    max_concurrent: int = 5,
    skip_if_exists: bool = True,
) -> list[int]:
    """Evaluate all benchmark responses in the database.

    Args:
        evaluation_method: "aggregate" for 3-metric scores, "fine_grained" for per-node/edge
        model: Evaluator model to use
        task_type: Filter responses by task type
        db_path: Path to SQLite database
        max_concurrent: Maximum concurrent evaluations
        skip_if_exists: Skip responses that already have evaluations

    Returns:
        List of response IDs that were evaluated
    """
    create_db(db_path)
    evaluator_type = f"llm_judge_{model}" if evaluation_method == "aggregate" else f"fine_grained_{model}"

    async with aiosqlite.connect(db_path) as conn:
        # Get responses to evaluate
        if skip_if_exists:
            cursor = await conn.execute(
                """
                SELECT br.id, br.response_answer, br.corrected_answer, cg.corrected_json, cg.input_text
                FROM benchmark_responses br
                JOIN causal_graphs cg ON br.sample_id = cg.sample_id
                WHERE br.task_type = ?
                AND NOT EXISTS (
                    SELECT 1 FROM benchmark_evaluations be
                    WHERE be.response_id = br.id AND be.evaluator_type = ?
                )
                """,
                (task_type, evaluator_type),
            )
        else:
            cursor = await conn.execute(
                """
                SELECT br.id, br.response_answer, br.corrected_answer, cg.corrected_json, cg.input_text
                FROM benchmark_responses br
                JOIN causal_graphs cg ON br.sample_id = cg.sample_id
                WHERE br.task_type = ?
                """,
                (task_type,),
            )

        rows = await cursor.fetchall()

    evaluated_ids: list[int] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_one(row: tuple) -> int | None:
        async with semaphore:
            response_id, response_answer, corrected_answer, ground_truth, input_text = row
            answer_to_evaluate = corrected_answer or response_answer

            if not answer_to_evaluate or not ground_truth:
                return None

            try:
                if evaluation_method == "aggregate":
                    await run_aggregate_evaluation(
                        response_id=response_id,
                        response_answer=answer_to_evaluate,
                        ground_truth=ground_truth,
                        input_text=input_text,
                        model=model,
                        db_path=db_path,
                    )
                else:
                    await run_fine_grained_evaluation(
                        response_id=response_id,
                        response_answer=answer_to_evaluate,
                        ground_truth=ground_truth,
                        input_text=input_text,
                        model=model,
                        db_path=db_path,
                    )
                return response_id
            except Exception as e:
                print(f"Error evaluating response {response_id}: {e}")
                return None

    tasks = [evaluate_one(row) for row in rows]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result is not None:
            evaluated_ids.append(result)

    return evaluated_ids


# Legacy function names for backwards compatibility
async def get_llm_judge_response(
    answer_id: int,
    model: str,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Get LLM judge evaluation for a benchmark response (legacy interface)."""
    async with aiosqlite.connect(db_path) as conn:
        # Try new schema first
        try:
            cursor = await conn.execute(
                """
                SELECT br.response_answer, br.corrected_answer, cg.corrected_json,
                       br.valid_format, br.task_type, cg.input_text
                FROM benchmark_responses br
                JOIN causal_graphs cg ON br.sample_id = cg.sample_id
                WHERE br.id = ?
                """,
                (answer_id,)
            )
        except:
            # Legacy schema fallback
            cursor = await conn.execute(
                """
                SELECT br.response_answer, br.corrected_answer, cg.corrected_json,
                       br.valid_format, br.evaluation_type, cg.md_unprocessed
                FROM benchmark_responses br
                JOIN causal_graphs cg ON br.causal_graph_id = cg.id
                WHERE br.id = ?
                """,
                (answer_id,)
            )

        row = await cursor.fetchone()

        if row and row[4] == "causal_graph_generation_with_node_names":
            raise RuntimeError("LLM judge evaluation not supported for causal_graph_generation_with_node_names")

    if not row:
        raise RuntimeError(f"No benchmark response found with ID {answer_id}")

    response_answer, corrected_answer, ground_truth, valid_format, _, input_text = row
    answer_to_evaluate = response_answer if valid_format else corrected_answer

    system_prompt = load_prompt("llm_judge").strip()
    user_message = (
        f"# Original Text\n\n{input_text}\n\n"
        f"# Ground Truth Causal Graph\n\n{ground_truth}\n\n"
        f"# Generated Causal Graph\n\n{answer_to_evaluate}"
    )

    result = await call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        include_reasoning=True,
    )

    if not result.answer:
        raise RuntimeError("No answer returned from LLM")

    if not verify_aggregate_output_format(result.answer):
        raise RuntimeError("Invalid output format returned from LLM")

    # Store evaluation in database
    create_db(db_path)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            INSERT INTO benchmark_evaluations
            (response_id, evaluation_date, evaluator_type, evaluation_answer, evaluation_reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (answer_id, datetime.now().isoformat(), f"llm_judge_{model}", result.answer, result.reasoning),
        )
        await conn.commit()

    return result.answer
