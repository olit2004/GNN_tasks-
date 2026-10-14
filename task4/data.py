import torch
from task1.generate_data import make_dataset

def get_data(n_total=1000, n_seen=800, seed=0):
    raw = make_dataset(n=n_total, n_shortcuts=100, seed=seed)
    X, edges, y = raw["X"], raw["edges"], raw["y"]

    seen_mask = (edges[:, 0] < n_seen) & (edges[:, 1] < n_seen)
    edges_seen = edges[seen_mask]
    X_seen, y_seen = X[:n_seen], y[:n_seen]

    mean, std = X_seen.mean(0), X_seen.std(0)
    scaler = {"mean": mean.tolist(), "std": std.tolist()}
    X_seen_norm = (X_seen - mean) / std

    tr = raw["train_mask"][:n_seen]
    va = raw["val_mask"][:n_seen]
    te = raw["test_mask"][:n_seen]

    X_unseen_norm = (X[n_seen:] - mean) / std

    return {
        "X_seen": X_seen_norm,
        "edges_seen": edges_seen,
        "y_seen": y_seen,
        "tr": tr,
        "va": va,
        "te": te,
        "scaler": scaler,
        "X_unseen": X_unseen_norm,
        "y_unseen": y[n_seen:],
        "all_edges": edges,
        "n_seen": n_seen,
        "n_unseen": n_total - n_seen,
    }
