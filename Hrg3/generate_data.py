
import torch

# Reproducibility
torch.manual_seed(20261013)

N = 600

# 1. Generate hidden node signals and nuisance variables
z = torch.randn(N)
u = torch.randn(N)

# 2. Generate three noisy numerical features per node
X = torch.stack([
    z + 0.9 * torch.randn(N),
    0.5 * z + 0.8 * u + 0.9 * torch.randn(N),
    -0.25 * z + 0.7 * u + torch.randn(N)
], dim=1)

X = (X - X.mean(0)) / X.std(0)

# 3. Generate graph connections
# Nodes with nearby indices form relay connections.
A = torch.zeros(N, N)

for i in range(N):
    for k in (1, 2):
        j = (i + k) % N
        A[i, j] = A[j, i] = 1

# Add random shortcut connections
for _ in range(180):
    i, j = torch.randint(N, (2,)).tolist()
    if i != j:
        A[i, j] = A[j, i] = 1

# Add misleading connections between opposite-sign signals
for _ in range(180):
    i, j = torch.randint(N, (2,)).tolist()
    if i != j and z[i] * z[j] < 0:
        A[i, j] = A[j, i] = 1

# 4. Generate labels from node and neighbor signals
neighbor_signal = A @ z / A.sum(1).clamp(min=1)
score = 0.85 * z + 0.65 * neighbor_signal + 0.45 * torch.randn(N)
y = (score > score.median()).long()

# 5. Split labeled nodes: 60% train, 20% validation, 20% test
train_mask = torch.zeros(N, dtype=torch.bool)
val_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)

for c in (0, 1):
    idx = torch.where(y == c)[0]
    idx = idx[torch.randperm(len(idx))]

    n_train = int(0.6 * len(idx))
    n_val = int(0.2 * len(idx))

    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True

# 6. Validate the dataset
assert X.shape == (600, 3)
assert y.shape == (600,)
assert set(y.tolist()) == {0, 1}
assert torch.allclose(A, A.T)
assert torch.all(torch.diag(A) == 0)
assert torch.all(train_mask | val_mask | test_mask)
assert not torch.any(train_mask & val_mask)
assert not torch.any(train_mask & test_mask)
assert not torch.any(val_mask & test_mask)

print("Dataset validated!")
print("Features:", X.shape)
print("Edges:", int(torch.triu(A, diagonal=1).sum()))
print("Class counts:", torch.bincount(y).tolist())
print("Train / validation / test:",
      train_mask.sum().item(),
      val_mask.sum().item(),
      test_mask.sum().item())

# Save the dataset
torch.save({
    "X": X,
    "A": A,
    "y": y,
    "train_mask": train_mask,
    "val_mask": val_mask,
    "test_mask": test_mask
}, "hrg3.pt")
