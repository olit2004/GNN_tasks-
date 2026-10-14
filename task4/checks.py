import json
import torch
import torch.nn.functional as F
from task1.model import GCN, normalize_adj
from task1.generate_data import adjacency
from task4.registry import PROD_DIR

def run_failure_tests(client):
    # 1. Isolated node (degree 0)
    res = client.post("/predict", json={"features": {0: [0,0,0], 1: [1,1,1]}, "edges": [], "target_ids": [0]}).json()
    assert res["predictions"][0]["isolated"] is True
    print("  . Isolated node: Handled via self-loop ego-features.")

    # 2. Duplicate target IDs
    res = client.post("/predict", json={"features": {0: [0,0,0], 1: [1,1,1]}, "edges": [[0,1]], "target_ids": [0, 0]}).json()
    assert len(res["predictions"]) == 1
    print("   Duplicate target IDs: Transparently deduplicated.")

    # 3. Invalid edge endpoint
    res = client.post("/predict", json={"features": {0: [0,0,0]}, "edges": [[0, 999]], "target_ids": [0]})
    assert res.status_code == 422
    print("   Invalid edge endpoint: Rejected with HTTP 422.")

    # 4. Incorrect feature dimension
    res = client.post("/predict", json={"features": {0: [0,0]}, "edges": [], "target_ids": [0]})
    assert res.status_code == 422
    print("   Incorrect feature dimension: Rejected with HTTP 422.")

    # 5. Unknown target ID
    res = client.post("/predict", json={"features": {0: [0,0,0]}, "edges": [], "target_ids": [55]})
    assert res.status_code == 422
    print("      Unknown target ID: Rejected with HTTP 422.")

    # 6. Model loading failure
    p_meta = PROD_DIR / "meta.json"
    p_meta.rename(PROD_DIR / "meta.json.bak")
    res = client.get("/health")
    assert res.status_code == 503
    (PROD_DIR / "meta.json.bak").rename(p_meta)
    print("  Model loading failure: Service returns HTTP 503.")

def check_consistency(client, data):
    sample_ids = list(range(10))
    with open(PROD_DIR / "meta.json") as f:
        meta = json.load(f)
    model = GCN(d_in=3, d_hidden=meta["hidden_dim"], d_out=2)
    model.load_state_dict(torch.load(PROD_DIR / "weights.pt", weights_only=True))
    model.eval()

    sub_e = [e for e in data["edges_seen"].tolist() if e[0] in sample_ids and e[1] in sample_ids]
    X_sub = data["X_seen"][:10]
    A_sub = normalize_adj(adjacency(torch.tensor(sub_e), 10))
    prob_offline = F.softmax(model(X_sub, A_sub), 1)[:, 1].tolist()

    raw_mean = torch.tensor(data["scaler"]["mean"])
    raw_std = torch.tensor(data["scaler"]["std"])
    X_raw = (X_sub * raw_std + raw_mean).tolist()

    api_res = client.post("/predict", json={
        "features": {i: X_raw[i] for i in range(10)},
        "edges": sub_e,
        "target_ids": sample_ids
    }).json()

    diffs = [abs(api_res["predictions"][i]["prob_c1"] - prob_offline[i]) for i in range(10)]
    assert max(diffs) < 1e-3
    print(f"  Max probability difference: {max(diffs):.2e}  ->  ✓ Offline and API match identically.")

def run_serving_demo(client, data):
    raw_mean = torch.tensor(data["scaler"]["mean"])
    raw_std = torch.tensor(data["scaler"]["std"])
    X_raw = (data["X_seen"][:3] * raw_std + raw_mean).tolist()

    # Existing nodes
    r_seen = client.post("/predict", json={
        "features": {i: X_raw[i] for i in range(3)}, "edges": [[0,1]], "target_ids": [0, 1]
    }).json()
    print(f"  Existing Node 0 -> {r_seen['predictions'][0]['label']} (prob {r_seen['predictions'][0]['prob_c1']})")

    # Unseen nodes
    unseen_raw = (data["X_unseen"][:2] * raw_std + raw_mean).tolist()
    r_unseen = client.post("/predict", json={
        "features": {800: unseen_raw[0], 801: unseen_raw[1], 0: X_raw[0]},
        "edges": [[800, 0], [801, 0]],
        "target_ids": [800, 801]
    }).json()
    print(f"  Unseen Node 800 -> {r_unseen['predictions'][0]['label']} (prob {r_unseen['predictions'][0]['prob_c1']})")
    print(f"  Unseen Node 801 -> {r_unseen['predictions'][1]['label']} (prob {r_unseen['predictions'][1]['prob_c1']})")
    print("  * Note: 2-layer GCN requires 2-hop neighborhood context. Unseen nodes with no edges fallback to self-loops.")
