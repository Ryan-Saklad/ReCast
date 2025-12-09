"""Benchmark runner for ReCast causal graph generation evaluation."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import aiosqlite

from .helpers import load_prompt, nodes_to_node_dict, create_db
from .openrouter import call_openrouter, call_openrouter_with_prefill

DEFAULT_DB_PATH = "recast_results.db"


def prepare_causal_graph_prompt(nodes: set[str], provide_node_names: bool = False) -> str:
    """Prepare the causal graph generation prompt."""
    if provide_node_names:
        template = load_prompt("causal_graph_generation_with_node_names")
        node_dict = nodes_to_node_dict(nodes)
        return template.replace("NODE_JSON", json.dumps(node_dict))
    else:
        template = load_prompt("causal_graph_generation")
        return template.replace("NUM_NODES", str(len(nodes)))


async def generate_causal_graph(
    text: str,
    nodes: set[str],
    model: str,
    provide_node_names: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 100000,
) -> tuple[str | None, str | None, str]:
    """Generate a causal graph for the given text using the specified model.

    Returns:
        Tuple of (answer, reasoning, finish_reason)
    """
    prompt = prepare_causal_graph_prompt(nodes, provide_node_names)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "<think>"},
    ]

    result = await call_openrouter_with_prefill(messages, model, temperature, max_tokens)
    return result.answer, result.reasoning, result.finish_reason


def process_answer(answer: str | None) -> str | None:
    """Clean up model answer by removing markdown wrappers."""
    if not answer:
        return None

    answer = answer.strip()
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{") : -1]
    if answer.startswith("```json") and answer.endswith("```"):
        answer = answer[len("```json") : -len("```")]

    return answer


def validate_graph_response(
    answer: str | None,
    provide_node_names: bool = False,
    num_nodes: int | None = None,
) -> bool:
    """Check if the model response is valid JSON with correct structure."""
    processed = process_answer(answer)
    if not processed:
        return False

    try:
        data = json.loads(processed)
        if "relationships" not in data:
            return False

        relationships = data["relationships"]
        if not isinstance(relationships, list):
            return False

        for rel in relationships:
            if not isinstance(rel, dict) or "source" not in rel or "sink" not in rel:
                return False
            if provide_node_names:
                if not isinstance(rel["source"], int) or not isinstance(rel["sink"], int):
                    return False
                if num_nodes and not (1 <= rel["source"] <= num_nodes and 1 <= rel["sink"] <= num_nodes):
                    return False

        return True
    except json.JSONDecodeError:
        return False


async def insert_response_to_db(
    sample_id: int,
    answer: str | None,
    reasoning: str | None,
    model: str,
    task_type: str,
    valid_format: bool,
    finish_reason: str | None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Insert a benchmark response into the database and return its ID."""
    create_db(db_path)

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO benchmark_responses
            (sample_id, response_date, task_type, model, response_reasoning, response_answer, finish_reason, valid_format)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sample_id, datetime.now().isoformat(), task_type, model, reasoning, answer, finish_reason, int(valid_format)),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore


async def correct_causal_graph(
    answer_id: int,
    model: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """Correct a malformed causal graph response using an LLM.

    Args:
        answer_id: ID of the benchmark response to correct
        model: Model to use for correction
        db_path: Path to SQLite database

    Returns:
        Corrected JSON dict or None if correction failed
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT response_reasoning, response_answer, valid_format, task_type FROM benchmark_responses WHERE id = ?",
            (answer_id,)
        )
        result = await cursor.fetchone()
        if not result or (result[0] is None and result[1] is None):
            raise ValueError(f"No benchmark response found with ID {answer_id}")

    reasoning, answer, valid_format, task_type = result
    is_valid = valid_format == 1
    nodes_provided = task_type == "causal_graph_generation_with_node_names"

    if is_valid:
        return None

    response = (reasoning or "") + (answer or "")

    if nodes_provided:
        # Get the node names from ground truth
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                """
                SELECT cg.corrected_json
                FROM benchmark_responses br
                JOIN causal_graphs cg ON br.sample_id = cg.sample_id
                WHERE br.id = ?
                """,
                (answer_id,)
            )
            result = await cursor.fetchone()
            if not result:
                raise ValueError(f"No causal graph found for benchmark response ID {answer_id}")

            corrected_json = result[0]
            corrected_data: list[dict[str, str]] = json.loads(corrected_json)
            nodes = {rel["source"] for rel in corrected_data} | {rel["sink"] for rel in corrected_data}

        system_prompt = load_prompt("correct_causal_graph_with_nodes")
        node_dict = nodes_to_node_dict(nodes)
        system_prompt = system_prompt.replace("NODE_JSON", json.dumps(node_dict))
    else:
        system_prompt = load_prompt("correct_causal_graph")

    # Truncate response to characters after index 5000 (matches original implementation)
    if len(response) > 5000:
        response = response[5000:]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": response}
    ]

    try:
        result = await call_openrouter(messages, model, max_tokens=10000, include_reasoning=False)

        if result.finish_reason != "stop":
            raise RuntimeError(f"Model finished with reason {result.finish_reason}")

        corrected_json_str = (result.answer or "").strip()

        # Remove markdown wrapper
        if corrected_json_str.startswith("```json"):
            corrected_json_str = corrected_json_str[len("```json"):]
        if corrected_json_str.endswith("```"):
            corrected_json_str = corrected_json_str[:-len("```")]

        corrected_data = json.loads(corrected_json_str)

        # Remove duplicate relationships
        seen = set()
        unique_relationships = []
        for rel in corrected_data["relationships"]:
            rel_tuple = (rel["source"], rel["sink"])
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                unique_relationships.append(rel)

        corrected_data["relationships"] = unique_relationships

        # Update the database
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE benchmark_responses SET corrected_answer = ? WHERE id = ?",
                (json.dumps(corrected_data), answer_id)
            )
            await db.commit()

        return corrected_data

    except Exception as e:
        print(f"Error correcting causal graph for ID {answer_id}: {e}")
        return None


async def run_benchmark_for_sample(
    sample_id: int,
    input_text: str,
    ground_truth_edges: list[dict[str, str]],
    model: str,
    provide_node_names: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 100000,
    db_path: str = DEFAULT_DB_PATH,
    correction_model: str | None = None,
    skip_if_exists: bool = True,
) -> int | None:
    """Run the benchmark for a single sample.

    Args:
        sample_id: Unique identifier for this sample
        input_text: The text to generate a causal graph from
        ground_truth_edges: List of edges with 'source' and 'sink'/'target' keys
        model: Model identifier (e.g., "deepseek/deepseek-r1")
        provide_node_names: Whether to provide node names in the prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        db_path: Path to SQLite database
        correction_model: If provided, use this model to correct malformed responses
        skip_if_exists: If True, skip if response already exists for this sample/model/type

    Returns:
        The benchmark response ID, or None if failed
    """
    nodes = {
        node for edge in ground_truth_edges
        for node in (edge["source"], edge.get("sink") or edge.get("target"))
    }
    task_type = "causal_graph_generation_with_node_names" if provide_node_names else "causal_graph_generation"

    # Check if response already exists
    if skip_if_exists:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                """
                SELECT id FROM benchmark_responses
                WHERE sample_id = ? AND model = ? AND task_type = ?
                """,
                (sample_id, model, task_type),
            )
            existing = await cursor.fetchone()
            if existing:
                return existing[0]

    try:
        answer, reasoning, finish_reason = await generate_causal_graph(
            input_text, nodes, model, provide_node_names, temperature, max_tokens
        )
        valid = validate_graph_response(answer, provide_node_names, len(nodes))

        response_id = await insert_response_to_db(
            sample_id, answer, reasoning, model, task_type, valid, finish_reason, db_path
        )

        # Auto-correct if invalid and correction_model is provided
        if not valid and correction_model:
            print(f"Response {response_id} has invalid format, attempting correction...")
            corrected = await correct_causal_graph(response_id, correction_model, db_path)
            if corrected:
                print(f"Successfully corrected response {response_id}")

        return response_id
    except Exception as e:
        print(f"Error generating causal graph for sample {sample_id}: {e}")
        return None


async def run_benchmark(
    samples: list[dict[str, Any]],
    model: str,
    provide_node_names: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 100000,
    db_path: str = DEFAULT_DB_PATH,
    max_concurrent: int = 10,
    correction_model: str | None = None,
    skip_if_exists: bool = True,
) -> list[int | None]:
    """Run the benchmark for multiple samples.

    Args:
        samples: List of dicts with 'id', 'input_text', and 'edges' keys
        model: Model identifier (e.g., "deepseek/deepseek-r1")
        provide_node_names: Whether to provide node names in prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        db_path: Path to SQLite database
        max_concurrent: Maximum concurrent API calls
        correction_model: If provided, use this model to correct malformed responses
        skip_if_exists: If True, skip samples that already have responses

    Returns:
        List of benchmark response IDs (or None for failures)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(sample: dict[str, Any]) -> int | None:
        async with semaphore:
            return await run_benchmark_for_sample(
                sample["id"],
                sample["input_text"],
                sample["edges"],
                model,
                provide_node_names,
                temperature,
                max_tokens,
                db_path,
                correction_model,
                skip_if_exists,
            )

    return await asyncio.gather(*[run_with_semaphore(s) for s in samples])
