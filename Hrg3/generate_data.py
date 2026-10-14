

import torch
from pathlib import Path


def adjacency(edges, n):
    """Dense symmetric 0/1 adjacency (no self loops) from an (E,2) edge list."""
    A = torch.zeros(n, n)
    A[edges[:, 0], edges[:, 1]] = 1.0
    A[edges[:, 1], edges[:, 0]] = 1.0
    return A


def make_dataset(n=1000, n_shortcuts=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)

    # 1. hidden signal + nuisance
    z, u = rn(n), rn(n)

    # 2. three noisy numerical features per node
    X = torch.stack([
        z + 1.0 * rn(n),                    # noisy copy of z
        z + 1.0 * rn(n),                    # another independent noisy copy
        0.5 * z + 0.8 * u + 0.7 * rn(n),    # weak copy mixed with nuisance
    ], dim=1)
    X = (X - X.mean(0)) / X.std(0)          # standardise (uses features only)

    # 3. connections: ring + random shortcuts (independent of z)
    pairs = set()
    for i in range(n):
        for k in (1, 2):
            j = (i + k) % n
            pairs.add((min(i, j), max(i, j)))
    target = len(pairs) + n_shortcuts
    while len(pairs) < target:
        i, j = torch.randint(n, (2,), generator=g).tolist()
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    edges = torch.tensor(sorted(pairs))
    A = adjacency(edges, n)

    # 4. labels depend mostly on the neighbours' hidden signal
    nbr_z = (A @ z) / A.sum(1).clamp(min=1)
    score = 0.4 * z + 1.2 * nbr_z + 0.25 * rn(n)
    y = (score > score.median()).long()

    # 5. stratified 60/20/20 split
    train, val, test = (torch.zeros(n, dtype=torch.bool) for _ in range(3))
    for c in (0, 1):
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        train[idx[:a]] = True
        val[idx[a:b]] = True
        test[idx[b:]] = True

    # 6. sanity checks
    assert X.shape == (n, 3) and set(y.tolist()) == {0, 1}
    assert torch.equal(A, A.T) and torch.all(torch.diag(A) == 0)
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert (train | val | test).all()

    return dict(X=X, y=y, edges=edges, train_mask=train, val_mask=val,
                test_mask=test, hidden_z=z)   # hidden_z: documentation only


def save_dataset(d, path="data/graph.pt"):
    Path(path).parent.mkdir(exist_ok=True)
    torch.save(d, path)


def load_dataset(path="data/graph.pt"):
    return torch.load(path)


def random_graph(n, n_edges, seed):
    """Random graph with the SAME number of edges (structure destroyed)."""
    g = torch.Generator().manual_seed(seed)
    pairs = set()
    while len(pairs) < n_edges:
        i, j = torch.randint(n, (2,), generator=g).tolist()
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return torch.tensor(sorted(pairs))


def remove_edges(edges, frac, seed):
    """Randomly drop a fraction of the edges."""
    g = torch.Generator().manual_seed(seed)
    keep = int(round(len(edges) * (1 - frac)))
    idx = torch.randperm(len(edges), generator=g)[:keep]
    return edges[idx]


def describe(d):
    X, y, edges = d["X"], d["y"], d["edges"]
    n = len(y)
    A = adjacency(edges, n)
    deg = A.sum(1)
    nbrX = (A @ X) / deg.clamp(min=1)[:, None]
    corr = lambda a, b: torch.corrcoef(torch.stack([a, b.float()]))[0, 1].item()
    print(f"Nodes {n} | features/node {X.shape[1]} | undirected edges {len(edges)}")
    print(f"Degree min/mean/max: {deg.min():.0f} / {deg.mean():.1f} / {deg.max():.0f}")
    print(f"Class counts: {torch.bincount(y).tolist()} | "
          f"train/val/test: {int(d['train_mask'].sum())}/"
          f"{int(d['val_mask'].sum())}/{int(d['test_mask'].sum())}")
    print("Correlation with label  (own feature  vs  mean of neighbours' feature):")
    for f in range(X.shape[1]):
        print(f"   feature {f}:  own {corr(X[:, f], y):+.3f}   neighbours {corr(nbrX[:, f], y):+.3f}")
    print("-> neighbours carry label information that the node's own features do not.")