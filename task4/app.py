import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from task1.model import GCN, normalize_adj
from task1.generate_data import adjacency

app = FastAPI(title="GCN Serving API")
PROD_DIR = Path("task4/production")

class PredictRequest(BaseModel):
    features: Dict[int, List[float]]
    edges: List[List[int]] = []
    target_ids: List[int]

def load_prod():
    if not (PROD_DIR / "weights.pt").exists() or not (PROD_DIR / "meta.json").exists():
        return None, None, None
    try:
        with open(PROD_DIR / "meta.json") as f:
            meta = json.load(f)
        model = GCN(d_in=3, d_hidden=meta["hidden_dim"], d_out=2)
        model.load_state_dict(torch.load(PROD_DIR / "weights.pt", weights_only=True))
        model.eval()
        return model, meta["scaler"], meta
    except Exception:
        return None, None, None

@app.get("/health")
def health():
    model, _, meta = load_prod()
    if model is None:
        raise HTTPException(503, "No production model loaded")
    return {"status": "healthy", "version": meta["version"]}

@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    model, scaler, meta = load_prod()
    if model is None:
        raise HTTPException(503, "Model unavailable")

    node_ids = sorted(req.features.keys())
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    # Input validations
    for nid, vec in req.features.items():
        if len(vec) != 3:
            raise HTTPException(422, f"Node {nid} feature length is {len(vec)}, expected 3")
    for u, v in req.edges:
        if u not in id_map or v not in id_map:
            raise HTTPException(422, f"Invalid edge ({u}, {v}): endpoint not in features")
    for tid in req.target_ids:
        if tid not in id_map:
            raise HTTPException(422, f"Target ID {tid} not found in features")

    # Graph construction & normalization
    A = torch.zeros(len(node_ids), len(node_ids))
    degs = {nid: 0 for nid in node_ids}
    for u, v in req.edges:
        A[id_map[u], id_map[v]] = A[id_map[v], id_map[u]] = 1.0
        degs[u] += 1
        degs[v] += 1
    A_hat = normalize_adj(A)

    # Standardize & forward pass
    X_raw = torch.tensor([req.features[nid] for nid in node_ids], dtype=torch.float32)
    X = (X_raw - torch.tensor(scaler["mean"])) / torch.tensor(scaler["std"])

    with torch.no_grad():
        logits = model(X, A_hat)
        probs = F.softmax(logits, dim=1)

    preds = []
    seen = set()
    for tid in req.target_ids:
        if tid in seen:
            continue
        seen.add(tid)
        idx = id_map[tid]
        c = int(logits[idx].argmax().item())
        preds.append({
            "node_id": tid,
            "class": c,
            "label": f"Class {c}",
            "prob_c1": round(float(probs[idx, 1]), 4),
            "isolated": (degs[tid] == 0)
        })

    return {
        "version": meta["version"],
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "predictions": preds,
    }
