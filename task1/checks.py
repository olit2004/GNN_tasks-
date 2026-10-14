"""Correctness checks."""
import torch
from data import adjacency
from model import normalize_adj, GCN, MLP


def check_manual_node(layer, X, edges, node=None):
    """Recompute ONE node's layer-1 output with explicit python loops + formula
        h_v = sum_{u in N(v) U {v}}  1/sqrt(d_v d_u) * (x_u W) + b
    and compare with the matrix implementation  (A_hat @ (X W) + b)[v]."""
    n = X.shape[0]
    nbrs = {i: set() for i in range(n)}
    for a, b in edges.tolist():
        nbrs[a].add(b); nbrs[b].add(a)
    if node is None:
        node = 5
    deg = {i: len(nbrs[i]) + 1 for i in range(n)}          # +1 = self-loop
    W, b = layer.weight.detach().double(), layer.bias.detach().double()
    Xd = X.double()

    print(f"Node {node}: neighbours = {sorted(nbrs[node])}  (+ itself via self-loop)")
    print(f"degree with self-loop d_v = {deg[node]}")
    h = torch.zeros(W.shape[1], dtype=torch.float64)
    for u in sorted(nbrs[node] | {node}):
        coef = 1.0 / (deg[node] * deg[u]) ** 0.5
        print(f"   u={u:4d}  d_u={deg[u]}  coef=1/sqrt({deg[node]}*{deg[u]})={coef:.4f}")
        h += coef * (Xd[u] @ W)
    h += b

    with torch.no_grad():
        full = layer(X, normalize_adj(adjacency(edges, n)))[node].double()
    diff = (h - full).abs().max().item()
    print(f"manual  : {[round(v, 6) for v in h.tolist()]}")
    print(f"matrix  : {[round(v, 6) for v in full.tolist()]}")
    print(f"max |difference| = {diff:.2e}  ->  {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


def check_permutation(model, X, edges, seed=123):
    """Relabel nodes with a random permutation (features AND edges), run the model,
    and check output_new == output_old[perm]."""
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)     # new position k holds old node perm[k]
    inv = torch.empty_like(perm); inv[perm] = torch.arange(n)   # old id -> new id
    X_p = X[perm]
    edges_p = inv[edges]                      # every edge endpoint renamed
    model.eval()
    with torch.no_grad():
        out = model(X, normalize_adj(adjacency(edges, n)))
        out_p = model(X_p, normalize_adj(adjacency(edges_p, n)))
    diff = (out_p - out[perm]).abs().max().item()
    same_pred = torch.equal(out_p.argmax(1), out[perm].argmax(1))
    print(f"max |f(permuted) - permuted(f)| = {diff:.2e} | identical predictions: {same_pred}"
          f"  ->  {'PASS' if diff < 1e-5 and same_pred else 'FAIL'}")
    return diff < 1e-5 and same_pred


def check_gcn_without_edges_is_mlp(gcn, X):
    """With no edges, A_hat = I, so the GCN must equal an MLP with the same weights."""
    mlp = MLP(X.shape[1], gcn.l1.weight.shape[1], gcn.l2.weight.shape[1])
    with torch.no_grad():
        mlp.fc1.weight.copy_(gcn.l1.weight.T); mlp.fc1.bias.copy_(gcn.l1.bias)
        mlp.fc2.weight.copy_(gcn.l2.weight.T); mlp.fc2.bias.copy_(gcn.l2.bias)
    gcn.eval(); mlp.eval()
    I = normalize_adj(torch.zeros(X.shape[0], X.shape[0]))
    with torch.no_grad():
        diff = (gcn(X, I) - mlp(X)).abs().max().item()
    print(f"max |GCN(no edges) - MLP(same weights)| = {diff:.2e}  ->  {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5