"""TEA-GIM graph similarity evaluation using pretrained GraphSAGE embeddings.

This is a separate experiment from the main benchmark evaluation.
Requires: pip install recast-bench[teagim]
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_add_pool

from .helpers import nodes_to_node_dict

DEFAULT_DB_PATH = "recast_results.db"
DEFAULT_MODEL_PATH = "models/GraphSAGE_arxiv_1000_tp.pth"
MODEL_URL = "https://huggingface.co/W-rudder/TEA-GLM/resolve/main/gnn/GraphSAGE_arxiv_1000_tp.pth"


class GraphSAGE(torch.nn.Module):
    """GraphSAGE model from TEA-GLM for graph embedding."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 2048,
        out_channels: int = 4096,
        n_layers: int = 2,
        dropout: float = 0.5,
        pooling: str = "sum",
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.activation = F.relu

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.fc1 = torch.nn.Linear(out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)

        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            from torch_geometric.nn import global_mean_pool
            self.pool = global_mean_pool
        elif pooling == "max":
            from torch_geometric.nn import global_max_pool
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """Used during contrastive training - not needed for similarity."""
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def download_model_if_missing(model_path: str = DEFAULT_MODEL_PATH) -> None:
    """Download the pretrained TEA-GLM GraphSAGE model if not present."""
    if not os.path.exists(model_path):
        print("Downloading pretrained GraphSAGE model from Hugging Face...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urlretrieve(MODEL_URL, model_path)
        print("Download complete.")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> tuple[GraphSAGE, torch.device]:
    """Load the pretrained GraphSAGE model."""
    download_model_if_missing(model_path)
    device = get_device()

    model = GraphSAGE(
        in_channels=128,
        hidden_channels=2048,
        out_channels=4096,
        n_layers=2,
        dropout=0.5,
        pooling="sum",
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def parse_graph(graph_dict: dict, feature_dim: int = 128) -> Data:
    """Convert graph JSON into PyG Data object."""
    relationships = graph_dict.get("relationships", [])
    nodes = set()
    edges = []

    for rel in relationships:
        source = rel["source"]
        sink = rel.get("sink") or rel.get("target")
        nodes.add(source)
        nodes.add(sink)
        edges.append((source, sink))

    if not edges:
        raise ValueError("Graph has no edges")

    node_to_idx = {name: idx for idx, name in enumerate(sorted(nodes))}
    edge_index = torch.tensor(
        [[node_to_idx[src], node_to_idx[dst]] for src, dst in edges],
        dtype=torch.long,
    ).t().contiguous()

    num_nodes = len(node_to_idx)
    x = torch.randn((num_nodes, feature_dim), dtype=torch.float32)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, batch=batch)


def compute_graph_similarity(
    ground_truth: dict,
    generated: dict,
    model: GraphSAGE | None = None,
    device: torch.device | None = None,
) -> float:
    """Compute cosine similarity between two graph embeddings.

    Args:
        ground_truth: Ground truth graph dict with 'relationships' key
        generated: Generated graph dict with 'relationships' key
        model: Pretrained GraphSAGE model (loaded if not provided)
        device: Torch device (detected if not provided)

    Returns:
        Cosine similarity score between -1 and 1
    """
    if model is None or device is None:
        model, device = load_model()

    gt_graph = parse_graph(ground_truth).to(device)
    gen_graph = parse_graph(generated).to(device)

    with torch.no_grad():
        gt_emb = model(gt_graph.x, gt_graph.edge_index, gt_graph.batch)
        gen_emb = model(gen_graph.x, gen_graph.edge_index, gen_graph.batch)

    return F.cosine_similarity(gt_emb, gen_emb).item()


def evaluate_all_responses(db_path: str = DEFAULT_DB_PATH) -> None:
    """Evaluate all benchmark responses in the database using graph similarity.

    Results are stored in the benchmark_evaluations table.
    """
    from tqdm import tqdm

    model, device = load_model()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cg.id, cg.corrected_json, br.id, br.corrected_answer, br.response_answer, br.evaluation_type
        FROM causal_graphs cg
        JOIN benchmark_responses br ON br.causal_graph_id = cg.id
        WHERE cg.corrected_json IS NOT NULL
        AND (br.corrected_answer IS NOT NULL OR br.response_answer IS NOT NULL)
        AND br.evaluation_type IN ('causal_graph_generation', 'causal_graph_generation_with_node_names')
    """)
    rows = cursor.fetchall()
    conn.close()

    print(f"Found {len(rows)} graph pairs to compare")

    for row in tqdm(rows):
        graph_id, gt_json, benchmark_id, corrected_answer, response_answer, evaluation_type = row

        try:
            ground_truth = {"relationships": json.loads(gt_json)}
            model_response = corrected_answer or response_answer

            if not model_response:
                continue

            model_response = model_response.replace("```json", "").replace("```", "").strip()
            model_answer = json.loads(model_response)

            # Convert node IDs to names for with_node_names evaluation
            if evaluation_type == "causal_graph_generation_with_node_names":
                gt_variables = {
                    node for rel in ground_truth["relationships"]
                    for node in (rel["source"], rel["sink"])
                }
                node_dict = nodes_to_node_dict(gt_variables)
                id_to_name = {node["id"]: node["name"] for node in node_dict["nodes"]}

                for rel in model_answer.get("relationships", []):
                    if isinstance(rel["source"], int):
                        rel["source"] = id_to_name.get(rel["source"], str(rel["source"]))
                    if isinstance(rel["sink"], int):
                        rel["sink"] = id_to_name.get(rel["sink"], str(rel["sink"]))

            if not model_answer.get("relationships"):
                continue

            sim = compute_graph_similarity(ground_truth, model_answer, model, device)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO benchmark_evaluations
                (model_response_id, evaluation_date, evaluator_type, score)
                VALUES (?, ?, 'graph_similarity', ?)
            """, (benchmark_id, datetime.now().isoformat(), sim))
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error processing graph {graph_id} (response {benchmark_id}): {e}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO benchmark_evaluations
                (model_response_id, evaluation_date, evaluator_type, score, evaluation_reasoning)
                VALUES (?, ?, 'graph_similarity', 0.0, ?)
            """, (benchmark_id, datetime.now().isoformat(), str(e)))
            conn.commit()
            conn.close()


if __name__ == "__main__":
    evaluate_all_responses()
