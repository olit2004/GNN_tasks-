from pathlib import Path

import torch


data_path = Path(__file__).parent / "data" / "hrg3.pt"
data = torch.load(data_path, map_location="cpu", weights_only=True)

features = data["X"]
adjacency = data["A"]
labels = data["y"]

print(f"Dataset: {data_path}")
print(f"Nodes: {labels.numel()}")
print(f"Features per node: {features.shape[1]}")
print(f"Edges: {int(torch.triu(adjacency, diagonal=1).sum())}")
print(f"Class counts: {torch.bincount(labels).tolist()}")
print(
    "Train / validation / test:",
    data["train_mask"].sum().item(),
    data["val_mask"].sum().item(),
    data["test_mask"].sum().item(),
)

print("\nFirst 10 nodes (feature values and label):")
for node in range(min(10, labels.numel())):
    print(
        f"Node {node:>3}: "
        f"features={features[node].tolist()}, label={labels[node].item()}"
    )

print("\nNeighbors of the first 10 nodes:")
for node in range(min(10, adjacency.shape[0])):
    neighbors = torch.where(adjacency[node] > 0)[0].tolist()
    print(f"Node {node:>3}: {neighbors}")
