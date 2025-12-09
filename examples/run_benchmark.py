"""Example: Run the ReCast benchmark on a model via OpenRouter.

This example shows how to run your own benchmarking using the OpenRouter API.
Results are stored in a local SQLite database for later analysis.
"""

import asyncio
import os

from recast import load_dataset, run_benchmark_for_sample, create_db


async def main():
    # Ensure API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Create database (safe to call multiple times)
    db_path = "my_benchmark_results.db"
    create_db(db_path)
    print(f"Using database: {db_path}")

    # Load dataset
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples")

    # Run on first sample as demo
    sample = dataset[0]
    print(f"\nSample {sample.id}: {sample.title[:50]}...")
    print(f"  Nodes: {sample.num_nodes}, Edges: {sample.num_edges}")

    # Generate causal graph (uses OpenRouter API)
    response_id = await run_benchmark_for_sample(
        sample_id=sample.id,
        input_text=sample.input_text,
        ground_truth_edges=sample.edges,
        model="deepseek/deepseek-r1",
        db_path=db_path,
    )

    if response_id:
        print(f"  Response saved with ID: {response_id}")
    else:
        print("  Failed to generate response")


if __name__ == "__main__":
    asyncio.run(main())
