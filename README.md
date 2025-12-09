# ReCast: Real-world Causal Structure Inference Benchmark

A benchmark for evaluating LLM causal reasoning on real-world academic text.

**Paper:** [Can Large Language Models Infer Causal Relationships from Real-World Text?](https://arxiv.org/abs/2505.18931)

## Overview

ReCast evaluates whether LLMs can infer causal relationships from authentic academic texts. Unlike synthetic benchmarks with explicit causal statements, ReCast uses real papers where causal relationships are often implicit, requiring genuine reasoning to extract.

**Key findings:**
- Best model (DeepSeek-R1) achieves only **0.477 F1**
- Performance degrades significantly as relationships become less explicit
- Causal reasoning, not entity recognition, is the primary bottleneck

## Installation

```bash
git clone https://github.com/RyanSaklad/ReCast.git
cd ReCast
pip install -e .
```

For API-based benchmarking, set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

## Quick Start

```python
from recast import load_dataset

# Load the benchmark dataset from HuggingFace
dataset = load_dataset()
print(f"Loaded {len(dataset)} samples")  # 292 samples

# Access a sample
sample = dataset[0]
print(f"Title: {sample.title}")
print(f"Nodes: {sample.num_nodes}, Edges: {sample.num_edges}")
print(f"Explicitness: {sample.explicitness:.0%}")

# Filter by domain
medical = dataset.filter_by_domain("Medicine")

# Filter by difficulty (lower explicitness = harder)
hard_samples = dataset.filter_by_max_explicitness(0.3)
```

## Running Your Model

```python
from recast import load_dataset, load_prompt, calculate_graph_metrics

dataset = load_dataset()
prompt_template = load_prompt("causal_graph_generation")

for sample in dataset:
    # Prepare prompt
    prompt = prompt_template.replace("NUM_NODES", str(sample.num_nodes))

    # Call your model
    response = your_model(system=prompt, user=sample.input_text)

    # Parse response (should be JSON with "relationships" key)
    predicted_edges = parse_response(response)

    # Evaluate
    precision, recall, f1, shd, norm_shd = calculate_graph_metrics(
        sample.edges,
        predicted_edges
    )
    print(f"Sample {sample.id}: F1={f1:.3f}, SHD={shd:.0f}")
```

See `examples/` for complete working examples.

## CLI Usage

Run the full benchmark on any model via OpenRouter:

```bash
# Run benchmark on all 292 samples
recast run deepseek/deepseek-r1

# Run with node names provided (easier mode)
recast run deepseek/deepseek-r1 --with-nodes

# Run on specific samples
recast run deepseek/deepseek-r1 --samples 1,2,3,4,5

# View your local results (deterministic evaluation)
recast results

# Export results to JSON
recast export -o results.json
```

### Viewing Paper Results

The HuggingFace dataset includes pre-computed model responses and evaluations from the paper. View them without running any models:

```bash
# View LLM judge results (Table 3 - semantic evaluation)
recast paper-results --llm-judge

# View graph similarity results (Table 4 - deterministic)
recast paper-results
```

Example LLM judge output:
```
╭──────────────────────────────────┬──────────────┬──────────────┬────────────────┬─────────────┬─────╮
│ Model                            │ Causal Acc   │ Causal Rec   │ Semantic Sim   │ Composite   │   N │
├──────────────────────────────────┼──────────────┼──────────────┼────────────────┼─────────────┼─────┤
│ deepseek/deepseek-r1             │ 3.20±0.72    │ 2.53±0.65    │ 3.16±0.64      │ 0.592±0.116 │ 292 │
│ qwen/qwq-32b                     │ 3.07±0.78    │ 2.37±0.67    │ 3.00±0.73      │ 0.562±0.128 │ 291 │
│ openai/o3-mini                   │ 2.87±0.78    │ 2.24±0.60    │ 2.90±0.65      │ 0.534±0.117 │ 292 │
╰──────────────────────────────────┴──────────────┴──────────────┴────────────────┴─────────────┴─────╯
```

## Dataset

The dataset is hosted on HuggingFace: [RyanSaklad/ReCast](https://huggingface.co/datasets/RyanSaklad/ReCast)

| Statistic | Value |
|-----------|-------|
| Samples | 292 |
| Nodes per graph | 5-140 (mean: 25) |
| Edges per graph | 6-205 (mean: 37) |
| Sources | PLOS, MDPI |
| Domains | 6 (Medicine, Economics, Engineering, Education, Agriculture, Environment) |

Each sample includes:
- **input_text**: Markdown text from an academic paper
- **nodes**: Ground truth causal variables
- **edges**: Ground truth directed causal relationships
- **explicitness**: Proportion of nodes explicitly mentioned in text (higher = easier)
- **node_explicitness**: Per-node explicitness scores (1=explicit, 2=implicit, 3=absent)

## Benchmark Results

The paper uses two evaluation methods:

**LLM Judge (Table 3)** - Semantic evaluation using DeepSeek-R1 as judge. Scores causal accuracy, recall, and semantic similarity on 1-5 scale.

| Model | Composite | Causal Acc | Causal Rec | Semantic Sim |
|-------|-----------|------------|------------|--------------|
| DeepSeek-R1 | 0.592 | 3.20 | 2.53 | 3.16 |
| QwQ-32B | 0.562 | 3.07 | 2.37 | 3.00 |
| o3-mini | 0.534 | 2.87 | 2.24 | 2.90 |
| Qwen-32B | 0.524 | 2.82 | 2.18 | 2.87 |
| Llama-8B | 0.460 | 2.45 | 1.91 | 2.54 |

**Deterministic (Table 4)** - Exact string matching for node/edge comparison.

| Model | F1 | Node Precision | Node Recall | Edge Precision | Edge Recall |
|-------|-----|----------------|-------------|----------------|-------------|
| DeepSeek-R1 | 0.477 | 0.893 | 0.522 | 0.817 | 0.260 |
| QwQ-32B | 0.450 | 0.881 | 0.488 | 0.802 | 0.242 |
| o3-mini | 0.415 | 0.862 | 0.459 | 0.806 | 0.208 |
| Qwen-32B | 0.381 | 0.862 | 0.434 | 0.747 | 0.181 |
| Llama-8B | 0.302 | 0.827 | 0.359 | 0.677 | 0.125 |

## Project Structure

```
ReCast/
├── src/recast/
│   ├── dataset.py      # HuggingFace dataset loader
│   ├── helpers.py      # Metrics (calculate_graph_metrics, etc.)
│   ├── benchmark.py    # Benchmark runner (OpenRouter API)
│   ├── evaluator.py    # LLM-as-judge evaluation
│   └── cli.py          # Command-line interface
├── prompts/            # Prompt templates used in the paper
└── tests/
```

## Citation

```bibtex
@misc{saklad2025largelanguagemodelsinfer,
  title={Can Large Language Models Infer Causal Relationships from Real-World Text?},
  author={Ryan Saklad and Aman Chadha and Oleg Pavlov and Raha Moraffah},
  year={2025},
  eprint={2505.18931},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2505.18931}
}
```

## License

This work is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
