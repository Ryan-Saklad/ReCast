"""Example: View published benchmark results from the paper."""

from recast import get_paper_results


def main():
    # Get LLM judge results (Table 1 in paper)
    print("=== LLM Judge Results ===")
    llm_results = get_paper_results("llm_judge")

    for model, metrics in sorted(llm_results.items()):
        print(f"\n{model}:")
        print(f"  Causal Accuracy:     {metrics['causal_accuracy']['mean']:.2f} ± {metrics['causal_accuracy']['std']:.2f}")
        print(f"  Causal Recall:       {metrics['causal_recall']['mean']:.2f} ± {metrics['causal_recall']['std']:.2f}")
        print(f"  Semantic Similarity: {metrics['semantic_similarity']['mean']:.2f} ± {metrics['semantic_similarity']['std']:.2f}")
        print(f"  Composite:           {metrics['composite']['mean']:.2f} ± {metrics['composite']['std']:.2f}")
        print(f"  Samples: {metrics['n']}")

    # Get deterministic metrics (Table 2 in paper)
    print("\n\n=== Deterministic Metrics ===")
    det_results = get_paper_results("deterministic")

    for model, metrics in sorted(det_results.items()):
        print(f"\n{model}:")
        print(f"  Node Precision: {metrics['node_precision']['mean']:.3f} ± {metrics['node_precision']['std']:.3f}")
        print(f"  Node Recall:    {metrics['node_recall']['mean']:.3f} ± {metrics['node_recall']['std']:.3f}")
        print(f"  Edge Precision: {metrics['edge_precision']['mean']:.3f} ± {metrics['edge_precision']['std']:.3f}")
        print(f"  Edge Recall:    {metrics['edge_recall']['mean']:.3f} ± {metrics['edge_recall']['std']:.3f}")
        print(f"  F1:             {metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f}")
        print(f"  SHD:            {metrics['shd']['mean']:.1f} ± {metrics['shd']['std']:.1f}")

    # Get results for "with node names" task variant
    print("\n\n=== With Node Names Task ===")
    wnn_results = get_paper_results("deterministic", "causal_graph_generation_with_node_names")

    if wnn_results:
        for model, metrics in sorted(wnn_results.items()):
            print(f"\n{model}:")
            print(f"  F1:  {metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f}")
            print(f"  SHD: {metrics['shd']['mean']:.1f} ± {metrics['shd']['std']:.1f}")
    else:
        print("No results for this task type")


if __name__ == "__main__":
    main()
