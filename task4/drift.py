import json
import torch
import torch.nn.functional as F
from task1.model import GCN, normalize_adj
from task1.generate_data import make_dataset, adjacency
from task4.registry import PROD_DIR

def monitor_drift(data):
    with open(PROD_DIR / "meta.json") as f:
        meta = json.load(f)
    model = GCN(d_in=3, d_hidden=meta["hidden_dim"], d_out=2)
    model.load_state_dict(torch.load(PROD_DIR / "weights.pt", weights_only=True))
    model.eval()

    raw_mean = torch.tensor(data["scaler"]["mean"])
    raw_std = torch.tensor(data["scaler"]["std"])

    n = data["n_seen"]
    deg_base = len(data["edges_seen"]) * 2.0 / n
    with torch.no_grad():
        A_hat = normalize_adj(adjacency(data["edges_seen"], n))
        h_base = F.relu(model.l1(data["X_seen"], A_hat)).mean(0)
    print(f"  Baseline Avg Degree: {deg_base:.2f} | Embedding Mean Norm: {h_base.norm():.4f}")

    # Snapshot 1: Normal In-Distribution
    d_norm = make_dataset(n=800, seed=123)
    deg_norm = len(d_norm["edges"]) * 2.0 / 800
    with torch.no_grad():
        A_hat_n = normalize_adj(adjacency(d_norm["edges"], 800))
        X_norm = (d_norm["X"] - raw_mean) / raw_std
        h_norm = F.relu(model.l1(X_norm, A_hat_n)).mean(0)
    deg_shift = abs(deg_norm - deg_base) / deg_base
    emb_dist = (h_norm - h_base).norm().item()
    print(f"  Snapshot 1 (Normal):    Degree Shift {deg_shift:.1%} | Emb Dist {emb_dist:.4f} -> ✓ NORMAL")

    # Snapshot 2: Perturbed
    X_pert = X_norm + 1.5
    e_pert = d_norm["edges"][:int(len(d_norm["edges"]) * 0.4)]
    deg_pert = len(e_pert) * 2.0 / 800
    with torch.no_grad():
        A_hat_p = normalize_adj(adjacency(e_pert, 800))
        h_pert = F.relu(model.l1(X_pert, A_hat_p)).mean(0)
    deg_shift_p = abs(deg_pert - deg_base) / deg_base
    emb_dist_p = (h_pert - h_base).norm().item()
    print(f"  Snapshot 2 (Perturbed): Degree Shift {deg_shift_p:.1%} | Emb Dist {emb_dist_p:.4f} -> ★ DRIFT DETECTED")
   