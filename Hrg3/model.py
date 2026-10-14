"""GCN written with plain tensor ops (no torch_geometric / ready-made GNN layers)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A):
    """A_hat = D^-1/2 (A + I) D^-1/2   (self-loops + symmetric degree normalisation)."""
    A_tilde = A + torch.eye(A.shape[0])           # add self-loops
    deg = A_tilde.sum(1)                          # degree incl. self-loop (>= 1)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]


class GCNLayer(nn.Module):
    """One message-passing layer:  H' = A_hat @ (H @ W) + b."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, H, A_hat):
        return A_hat @ (H @ self.weight) + self.bias


class GCN(nn.Module):
    """Two layers:  logits = A_hat @ relu(A_hat X W1 + b1) @ W2 + b2."""
    def __init__(self, d_in=3, d_hidden=16, d_out=2, dropout=0.5):
        super().__init__()
        self.l1, self.l2 = GCNLayer(d_in, d_hidden), GCNLayer(d_hidden, d_out)
        self.dropout = dropout

    def forward(self, X, A_hat):
        h = F.relu(self.l1(X, A_hat))
        h = F.dropout(h, self.dropout, self.training)
        return self.l2(h, A_hat)


class MLP(nn.Module):
    """Same sizes as the GCN but never looks at the graph (A_hat is ignored)."""
    def __init__(self, d_in=3, d_hidden=16, d_out=2, dropout=0.5):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_out)
        self.dropout = dropout

    def forward(self, X, A_hat=None):
        h = F.relu(self.fc1(X))
        h = F.dropout(h, self.dropout, self.training)
        return self.fc2(h)