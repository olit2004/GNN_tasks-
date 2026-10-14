"""
Task 4: Complete GCN MLOps Pipeline Runner.
Run with: python task4/run.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from task4.app import app
from task4.data import get_data
from task4.registry import reset, train_and_track, evaluate_gate, promote, rollback
from task4.drift import monitor_drift
from task4.checks import run_failure_tests, check_consistency, run_serving_demo

client = TestClient(app)

def H(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def main():
    H("0. UNTRACKED BASELINE VS MLOPS PIPELINE")
    print("Untracked scripts suffer from 4 critical risks:")
    print("  No Lineage: Missing git commit and dataset version link.")
    print("  No Schema: Missing feature dimension and scaler mean/std.")
    print("  Blind Overwriting: New weights overwrite production with no validation gate.")
    print("  Lost Experiments: Terminal outputs disappear on exit.")

    H("1. DATASET: TRANSDUCTIVE VS INDUCTIVE SEPARATION")
    data = get_data(n_total=1000, n_seen=800, seed=0)
    print(f"Training Graph: {data['n_seen']} nodes | {len(data['edges_seen'])} edges")
    print(f"Held-out Unseen Graph: {data['n_unseen']} nodes (all incident edges excluded from training)")

    H("2 & 3. EXPERIMENT TRACKING & REGISTRY VALIDATION GATE")
    reset()

    # Exp 1: Baseline
    r1, m1 = train_and_track("baseline", "v1.0.0", 16, 0.01, 5e-4, 150, 42, data)
    ok, reason = evaluate_gate(m1)
    if ok: promote(r1, m1)
    print(f"Exp 1 (v1.0.0): Val Acc {m1['metrics']['val_acc']} | Test Acc {m1['metrics']['test_acc']} -> {reason}")

    # Exp 2: Tuned Champion (Beats on tie-breaker lower loss)
    r2, m2 = train_and_track("champion", "v1.1.0", 32, 0.01, 1e-4, 200, 42, data)
    ok, reason = evaluate_gate(m2)
    if ok: promote(r2, m2)
    print(f"Exp 2 (v1.1.0): Val Acc {m2['metrics']['val_acc']} | Test Acc {m2['metrics']['test_acc']} -> {reason}")

    # Exp 3: Degraded Regression (Blocked by gate)
    r3, m3 = train_and_track("regression", "v1.2.0-bad", 4, 0.8, 0.2, 20, 99, data)
    ok, reason = evaluate_gate(m3)
    if ok: promote(r3, m3)
    print(f"Exp 3 (v1.2.0-bad): Val Acc {m3['metrics']['val_acc']} -> {reason}")

    H(" ROBUST FAILURE HANDLING (6 TEST CASES)")
    run_failure_tests(client)

    H(" PREDICTION CONSISTENCY (OFFLINE VS API)")
    check_consistency(client, data)

    H("SERVING SETTINGS (EXISTING VS UNSEEN NODES)")
    run_serving_demo(client, data)

    H("DRIFT DETECTION (STRUCTURE & EMBEDDINGS)")
    monitor_drift(data)

    H(" ROLLBACK DEMONSTRATION")
    old_v, restored_v = rollback()
    print(f"  Current Production Model: {old_v}")
    print(f"  Successfully rolled back to: {restored_v}")

    print(f"\n{'='*65}\n  ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY!\n{'='*65}\n")

if __name__ == "__main__":
    main()
