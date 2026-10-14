"""Training loop: loss, backprop, weight update, validation-based model selection."""
import copy
import torch
import torch.nn.functional as F


def accuracy(logits, y, mask):
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def train(model, X, A_hat, y, masks, epochs=200, lr=0.01, weight_decay=5e-4,
          verbose=True, print_every=20, name="model"):
    """Full-batch training. Loss uses ONLY training labels.
    The checkpoint with the best validation accuracy is restored at the end;
    test accuracy is measured once, on that checkpoint."""
    tr, va, te = masks
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = dict(val_acc=-1.0, val_loss=1e9, epoch=0, state=None)

    if verbose:
        print(f"\n--- Training {name}: {epochs} epochs, Adam lr={lr}, wd={weight_decay} ---")
    for epoch in range(1, epochs + 1):
        # forward + loss on training nodes only
        model.train()
        opt.zero_grad()
        logits = model(X, A_hat)
        loss = F.cross_entropy(logits[tr], y[tr])
        # backward + weight update
        loss.backward()
        opt.step()

        # evaluate (no dropout, no gradients)
        model.eval()
        with torch.no_grad():
            out = model(X, A_hat)
            vl = F.cross_entropy(out[va], y[va]).item()
            tr_acc, va_acc = accuracy(out, y, tr), accuracy(out, y, va)
        if va_acc > best["val_acc"] or (va_acc == best["val_acc"] and vl < best["val_loss"]):
            best.update(val_acc=va_acc, val_loss=vl, epoch=epoch,
                        state=copy.deepcopy(model.state_dict()))
        if verbose and (epoch == 1 or epoch % print_every == 0):
            print(f"epoch {epoch:3d} | train loss {loss.item():.4f} acc {tr_acc:.3f} "
                  f"| val loss {vl:.4f} acc {va_acc:.3f}")

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        test_acc = accuracy(model(X, A_hat), y, te)
    if verbose:
        print(f"Best epoch {best['epoch']} (val acc {best['val_acc']:.3f})  ->  TEST acc {test_acc:.3f}")
    return dict(test_acc=test_acc, val_acc=best["val_acc"], best_epoch=best["epoch"])