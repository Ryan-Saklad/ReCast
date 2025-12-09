"""Example: Explore the ReCast dataset."""

from recast import load_dataset


def main():
    dataset = load_dataset()

    # Use built-in statistics() method
    print("=== Dataset Statistics ===")
    stats = dataset.statistics()

    print(f"\nTotal samples: {stats['total_samples']}")

    print(f"\nDomains:")
    for domain, count in sorted(stats['domains'].items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")

    print(f"\nSources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count}")

    print(f"\nPublication dates:")
    dr = stats['date_range']
    print(f"  Range: {dr['earliest']} to {dr['latest']}")
    print(f"  Samples with dates: {dr['samples_with_dates']}")

    print(f"\nGraph complexity:")
    print(f"  Nodes: {stats['nodes']['min']}-{stats['nodes']['max']} (avg {stats['nodes']['mean']:.1f})")
    print(f"  Edges: {stats['edges']['min']}-{stats['edges']['max']} (avg {stats['edges']['mean']:.1f})")

    print(f"\nExplicitness:")
    print(f"  Range: {stats['explicitness']['min']:.2f} - {stats['explicitness']['max']:.2f}")
    print(f"  Mean:  {stats['explicitness']['mean']:.2f}")

    # Filtering examples
    print("\n=== Filtering Examples ===")

    # By domain
    medical = dataset.filter_by_domain("Medicine")
    print(f"\nMedicine samples: {len(medical)}")

    # By source
    plos = dataset.filter_by_source("PLOS")
    print(f"PLOS samples: {len(plos)}")

    # By date (e.g., papers after knowledge cutoff)
    recent = dataset.filter_by_date_after("2023-01-01")
    print(f"Papers after 2023-01-01: {len(recent)}")

    # By explicitness
    easy = dataset.filter_by_min_explicitness(0.8)
    hard = dataset.filter_by_max_explicitness(0.2)
    print(f"High explicitness (>=0.8): {len(easy)}")
    print(f"Low explicitness (<=0.2): {len(hard)}")

    # By graph size
    small = dataset.filter_by_max_nodes(10)
    large = dataset.filter_by_min_nodes(50)
    print(f"Small graphs (<=10 nodes): {len(small)}")
    print(f"Large graphs (>=50 nodes): {len(large)}")

    # Chained filters
    hard_medical = dataset.filter_by_domain("Medicine").filter_by_max_explicitness(0.5)
    print(f"Hard medical samples: {len(hard_medical)}")

    sample_ids = dataset[:5].get_sample_ids() if hasattr(dataset[:5], 'get_sample_ids') else [s.id for s in dataset[:5]]
    print(f"\nFirst 5 sample IDs: {sample_ids}")


if __name__ == "__main__":
    main()
