"""
Master Demonstration Script for Task 4 MLOps Pipeline.
Runs the complete lifecycle in sequence:
0. Baseline comparison & limitations
1. Inductive data split & fingerprinting
2. Experiment tracking (3 runs) & model packaging
3. Registry evaluation gate (promotion & regression block)
4. Robust failure handling (6 cases)
5. Prediction consistency (Offline vs API)
6. Serving settings (Existing vs Unseen nodes)
7. Graph structure & embedding drift detection
8. Model rollback to previous approved version
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
from task4.pipeline.dataset import get_inductive_splits
from task4.pipeline.tracker import ExperimentTracker
from task4.pipeline.registry import ModelRegistry
from task4.pipeline.train_tracked import run_experiment
from task4.tests.test_failure_handling import (
    test_1_isolated_nodes, test_2_duplicate_target_ids, test_3_invalid_edge_endpoints,
    test_4_incorrect_feature_dimension, test_5_unknown_target_id, test_6_model_loading_failure
)
from task4.tests.test_prediction_consistency import test_prediction_consistency
from task4.tests.test_serving_settings import demonstrate_serving_settings
from task4.pipeline.drift_monitor import demonstrate_drift
from task4.pipeline.rollback import perform_rollback

def header(title):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)

def main():


    # ── 0. UNTRACKED BASELINE COMPARISON ──────────────────────────────────────
    header("0. Untracked Baseline & Limitations")
    print("""
Limitations of bare training scripts:
  1. No Lineage: Missing association between data split, git commit, and weights.
  2. No Metadata / Schema: Raw PyTorch tensors lack feature dimension and scaler mean/std.
  3. No Regression Gate: Scripts blindly overwrite production weights even when accuracy drops.
  4. Non-Reproducible: Terminal outputs disappear on exit.
    """)

    # ── 1. INDUCTIVE DATASET SPLIT ───────────────────────────────────────────
    header("1. Inductive Dataset Separation")
    data = get_inductive_splits(n_total=1000, n_seen=800, seed=0)
    print(f"  Total Nodes:       1,000")
    print(f"  Seen Nodes (Train/Val/Test):  {data['n_seen']} nodes | {len(data['edges_seen'])} edges")
    print(f"  Unseen Nodes (Held-out):      {data['n_unseen']} nodes (all incident edges excluded from training)")
    print(f"  Dataset Fingerprint:          {data['fingerprint']} (SHA-256)")

    # ── 2 & 3. EXPERIMENT TRACKING & REGISTRY PROMOTION GATE ──────────────────
    header("2 & 3. Experiment Tracking & Registry Promotion Gate")
    import shutil
    if Path("task4/registry/manifest.json").exists():
        Path("task4/registry/manifest.json").unlink()
    if Path("task4/registry/production").exists():
        shutil.rmtree("task4/registry/production")
    if Path("task4/registry/archived").exists():
        shutil.rmtree("task4/registry/archived")

    tracker = ExperimentTracker("gcn_production")
    registry = ModelRegistry()

    # Run 1: Baseline
    p1 = {"hidden_dim": 16, "lr": 0.01, "dropout": 0.0, "weight_decay": 5e-4, "epochs": 150, "seed": 42}
    c1, pass1 = run_experiment("exp1_baseline_gcn", p1, "v1.0.0", tracker, registry, data)

    # Run 2: Tuned Champion
    p2 = {"hidden_dim": 32, "lr": 0.01, "dropout": 0.2, "weight_decay": 1e-4, "epochs": 200, "seed": 42}
    c2, pass2 = run_experiment("exp2_tuned_champion", p2, "v1.1.0", tracker, registry, data)

    # Run 3: Regression Candidate (Should be BLOCKED)
    p3 = {"hidden_dim": 8, "lr": 0.25, "dropout": 0.5, "weight_decay": 1e-2, "epochs": 80, "seed": 99}
    c3, pass3 = run_experiment("exp3_degraded_regression", p3, "v1.2.0-bad", tracker, registry, data)

    current_prod = registry.get_production_model()
    print(f"\n  Active Approved Production Model: Version {current_prod['version']}")
    print(f"  Production Metrics: Val Acc={current_prod['metrics']['val_acc']:.4f} | Test Acc={current_prod['metrics']['test_acc']:.4f}")

    # ── 4. FAILURE HANDLING ──────────────────────────────────────────────────
    header("4. Robust Failure Handling (6 Test Cases)")
    test_1_isolated_nodes()
    test_2_duplicate_target_ids()
    test_3_invalid_edge_endpoints()
    test_4_incorrect_feature_dimension()
    test_5_unknown_target_id()
    test_6_model_loading_failure()

    # ── 5. PREDICTION CONSISTENCY ────────────────────────────────────────────
    header("5. Prediction Consistency (Offline vs API)")
    test_prediction_consistency()

    # ── 6. SERVING SETTINGS (TRANSDUCTIVE VS INDUCTIVE) ──────────────────────
    header("6. Serving Settings (Existing vs Unseen Nodes)")
    demonstrate_serving_settings()

    # ── 7. DRIFT DETECTION ───────────────────────────────────────────────────
    header("7. Drift Detection (Structure & Embeddings)")
    demonstrate_drift()

    # ── 8. ROLLBACK DEMONSTRATION ────────────────────────────────────────────
    header("8. Model Rollback Demonstration")
    print(f"  Before Rollback: Active Version = {registry.get_production_model()['version']}")
    restored = perform_rollback()
    print(f"  After Rollback:  Active Version = {registry.get_production_model()['version']}")
    assert restored["version"] == "v1.0.0", "Rollback did not restore expected version!"

    print("\n" + "=" * 70)
    print("  ★ ALL TASKS & PIPELINE DEMONSTRATIONS COMPLETED SUCCESSFULLY! ★")
    print("=" * 70)

if __name__ == "__main__":
    main()
