"""Run everything for Task 1:  python run_all.py"""
import torch
from data import make_dataset, save_dataset, describe, adjacency, random_graph, remove_edges
from model import GCN, MLP, normalize_adj
from train import train, accuracy
import checks

SEEDS = list(range(10))
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
bar = lambda v: "#" * int(round(v * 50))
ms = lambda xs: (torch.tensor(xs).mean().item(), torch.tensor(xs).std().item())
H = lambda t: print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def fit(kind, X, edges, y, masks, seed, verbose=False, name=""):
    torch.manual_seed(seed)
    model = GCN() if kind == "gcn" else MLP()
    A_hat = normalize_adj(adjacency(edges, len(y)))
    res = train(model, X, A_hat, y, masks, verbose=verbose, name=name or kind)
    return model, res


def main():
    d = make_dataset(n=1000, n_shortcuts=100, seed=0)
    save_dataset(d)
    X, y, edges = d["X"], d["y"], d["edges"]
    masks = (d["train_mask"], d["val_mask"], d["test_mask"])
    n = len(y)

    H("0. DATASET")
    describe(d)

    H("1. TRAINING ONE GCN (seed 0) - watch loss go down")
    gcn0, _ = fit("gcn", X, edges, y, masks, seed=0, verbose=True, name="GCN")
    fit("mlp", X, edges, y, masks, seed=0, verbose=True, name="MLP (no graph)")

    H("2. CORRECTNESS: one node, one layer - manual vs matrix")
    ok1 = checks.check_manual_node(gcn0.l1, X, edges, node=5)
    H("3. CORRECTNESS: permutation equivariance")
    ok2 = checks.check_permutation(gcn0, X, edges)
    H("4. CORRECTNESS: GCN with no edges == MLP")
    ok3 = checks.check_gcn_without_edges_is_mlp(gcn0, X)

    H(f"5. GCN vs MLP vs RANDOM GRAPH  ({len(SEEDS)} seeds, same split)")
    res = {"GCN (real graph)": [], "MLP (no graph)": [], "GCN (random graph)": []}
    trained = []
    for s in SEEDS:
        m, r1 = fit("gcn", X, edges, y, masks, s); trained.append(m)
        _, r2 = fit("mlp", X, edges, y, masks, s)
        _, r3 = fit("gcn", X, random_graph(n, len(edges), 1000 + s), y, masks, s)
        res["GCN (real graph)"].append(r1["test_acc"])
        res["MLP (no graph)"].append(r2["test_acc"])
        res["GCN (random graph)"].append(r3["test_acc"])
        print(f"seed {s}: GCN {r1['test_acc']:.3f} | MLP {r2['test_acc']:.3f} | random-graph GCN {r3['test_acc']:.3f}")
    print(f"\n{'model':22s} test accuracy (mean +- std)")
    for k, v in res.items():
        m, sd = ms(v)
        print(f"{k:22s} {m:.3f} +- {sd:.3f}  {bar(m)}")
    wins = sum(a > b for a, b in zip(res["GCN (real graph)"], res["MLP (no graph)"]))
    print(f"GCN beats MLP in {wins}/{len(SEEDS)} seeds")

    H("6. ABLATION A: retrain after removing edges")
    retrain = []
    for f in FRACS:
        accs = [fit("gcn", X, remove_edges(edges, f, 500 + s), y, masks, s)[1]["test_acc"] for s in SEEDS]
        retrain.append(ms(accs)[0])
        print(f"removed {int(f * 100):3d}% edges -> test {ms(accs)[0]:.3f} +- {ms(accs)[1]:.3f}  {bar(ms(accs)[0])}")

    H("7. ABLATION B: take the TRAINED GCN, damage the graph only at test time")
    infer = []
    for f in FRACS:
        accs = []
        for s, m in zip(SEEDS, trained):
            A_hat = normalize_adj(adjacency(remove_edges(edges, f, 500 + s), n))
            m.eval()
            with torch.no_grad():
                accs.append(accuracy(m(X, A_hat), y, masks[2]))
        infer.append(ms(accs)[0])
        print(f"removed {int(f * 100):3d}% edges -> test {ms(accs)[0]:.3f} +- {ms(accs)[1]:.3f}  {bar(ms(accs)[0])}")

    H("SUMMARY")
    print(f"checks passed: manual={ok1} permutation={ok2} no-edges==MLP={ok3}")
    for k, v in res.items():
        print(f"{k:22s} {ms(v)[0]:.3f} +- {ms(v)[1]:.3f}")
    print("If GCN > MLP > or ~ random-graph GCN and accuracy falls as edges are removed,")
    print("the model is really using the graph structure.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        names, vals = list(res), [ms(v) for v in res.values()]
        ax[0].bar(range(3), [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=4,
                  color=["#2a6fdb", "#888888", "#d9822b"])
        ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["GCN", "MLP", "GCN\nrandom graph"])
        ax[0].set_ylim(0.4, 1.0); ax[0].set_ylabel("test accuracy"); ax[0].set_title("Model comparison")
        pc = [100 * f for f in FRACS]
        ax[1].plot(pc, retrain, "o-", label="retrained on damaged graph", color="#2a6fdb")
        ax[1].plot(pc, infer, "s--", label="trained on full graph, damaged at test", color="#d9822b")
        ax[1].axhline(ms(res["MLP (no graph)"])[0], color="#888888", ls=":", label="MLP")
        ax[1].set_xlabel("% edges removed"); ax[1].set_ylim(0.4, 1.0)
        ax[1].set_title("Effect of removing edges"); ax[1].legend(fontsize=8)
        plt.tight_layout(); plt.savefig("results.png", dpi=150)
        print("Saved plot to results.png")
    except ImportError:
        print("(matplotlib not installed - run: pip install numpy matplotlib  to get results.png)")


if __name__ == "__main__":
    main()