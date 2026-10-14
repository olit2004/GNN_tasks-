from pathlib import Path
import json
import shutil
import torch
import torch.nn.functional as F
from task1.model import GCN, normalize_adj
from task1.generate_data import adjacency

PROD_DIR = Path("task4/production")
ARCHIVE_DIR = Path("task4/archive")
RUNS_DIR = Path("task4/runs")

def reset():
    if PROD_DIR.exists(): shutil.rmtree(PROD_DIR)
    if ARCHIVE_DIR.exists(): shutil.rmtree(ARCHIVE_DIR)
    if RUNS_DIR.exists(): shutil.rmtree(RUNS_DIR)

def train_and_track(name, version, hidden_dim, lr, wd, epochs, seed, data):
    torch.manual_seed(seed)
    model = GCN(d_in=3, d_hidden=hidden_dim, d_out=2)
    A_hat = normalize_adj(adjacency(data["edges_seen"], len(data["y_seen"])))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    X, y, tr, va, te = data["X_seen"], data["y_seen"], data["tr"], data["va"], data["te"]
    best_va_acc, best_va_loss, best_state = -1.0, 1e9, None

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(X, A_hat)[tr], y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(X, A_hat)
            acc = (out[va].argmax(1) == y[va]).float().mean().item()
            vl = F.cross_entropy(out[va], y[va]).item()
        if acc > best_va_acc or (acc == best_va_acc and vl < best_va_loss):
            best_va_acc, best_va_loss = acc, vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_acc = (model(X, A_hat)[te].argmax(1) == y[te]).float().mean().item()

    metrics = {"val_acc": round(best_va_acc, 4), "val_loss": round(best_va_loss, 4), "test_acc": round(te_acc, 4)}
    meta = {"version": version, "name": name, "hidden_dim": hidden_dim, "lr": lr, "scaler": data["scaler"], "metrics": metrics}

    run_dir = RUNS_DIR / version
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "weights.pt")
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Verification: reload check
    rm = GCN(d_in=3, d_hidden=hidden_dim, d_out=2)
    rm.load_state_dict(torch.load(run_dir / "weights.pt", weights_only=True))
    rm.eval()
    diff = (model(X, A_hat) - rm(X, A_hat)).abs().max().item()
    assert diff < 1e-7, "Packaging reload verification failed!"

    return run_dir, meta

def evaluate_gate(cand_meta):
    cand_acc, cand_loss = cand_meta["metrics"]["val_acc"], cand_meta["metrics"]["val_loss"]
    if not (PROD_DIR / "meta.json").exists():
        return True, "PROMOTED: First production model (initial baseline)."

    with open(PROD_DIR / "meta.json") as f:
        prod_meta = json.load(f)
    p_acc, p_loss = prod_meta["metrics"]["val_acc"], prod_meta["metrics"]["val_loss"]

    if cand_acc > p_acc:
        return True, f"PROMOTED: Candidate val_acc ({cand_acc}) beat production ({p_acc})."
    elif abs(cand_acc - p_acc) < 1e-6:
        if cand_loss < p_loss:
            return True, f"PROMOTED (TIE-BREAKER): Same accuracy, but candidate loss lower ({cand_loss} < {p_loss})."
        return False, f"REJECTED: Tied accuracy but loss not better ({cand_loss} >= {p_loss})."
    return False, f"REJECTED: Regression ({cand_acc} < {p_acc})."

def promote(run_dir, meta):
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if (PROD_DIR / "meta.json").exists():
        with open(PROD_DIR / "meta.json") as f:
            old_v = json.load(f)["version"]
        dest = ARCHIVE_DIR / old_v
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(PROD_DIR, dest)
    if PROD_DIR.exists(): shutil.rmtree(PROD_DIR)
    shutil.copytree(run_dir, PROD_DIR)

def rollback():
    with open(PROD_DIR / "meta.json") as f:
        cur_v = json.load(f)["version"]
    archived = sorted(ARCHIVE_DIR.iterdir())
    if not archived:
        raise RuntimeError("No archived models found for rollback.")
    target = archived[-1]
    shutil.rmtree(PROD_DIR)
    shutil.copytree(target, PROD_DIR)
    with open(PROD_DIR / "meta.json") as f:
        restored_v = json.load(f)["version"]
    return cur_v, restored_v
