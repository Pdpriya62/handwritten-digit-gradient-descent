"""Batch Gradient Descent training and the learning-rate sweep."""
import time

import numpy as np

from .model import SoftmaxClassifier


def train(data, lr, epochs=100, seed=0, verbose=False):
    """Train a SoftmaxClassifier with full-batch Gradient Descent.

    Update rule: W <- W - lr * dJ/dW,  b <- b - lr * dJ/db.
    Every run with the same `seed` starts from identical parameters, so runs that
    differ only in `lr` are directly comparable.

    history["train_loss"][t] is J measured at the start of epoch t (the value the
    gradient was computed from); validation metrics are measured after the update.
    """
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    model = SoftmaxClassifier(Xtr.shape[1], 10, seed=seed)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(epochs):
        J, dW, db = model.loss_and_grads(Xtr, ytr)
        model.W -= lr * dW
        model.b -= lr * db

        hist["train_loss"].append(J)
        hist["val_loss"].append(model.loss(Xva, yva))
        hist["train_acc"].append(model.accuracy(Xtr, ytr))
        hist["val_acc"].append(model.accuracy(Xva, yva))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  lr={lr:<7g} epoch {epoch + 1:>4}/{epochs}  "
                  f"loss {J:.4f}  val_acc {hist['val_acc'][-1]:.4f}")
    hist["seconds"] = time.time() - t0
    return model, hist


def lr_sweep(data, lrs, epochs=100, seed=0, verbose=False):
    """Train one model per learning rate. Returns {lr: (model, history)}."""
    results = {}
    for lr in lrs:
        results[lr] = train(data, lr, epochs=epochs, seed=seed, verbose=verbose)
    return results


def best_by_val_acc(results):
    """Return (lr, model, history) with the highest final validation accuracy."""
    lr = max(results, key=lambda k: results[k][1]["val_acc"][-1])
    model, hist = results[lr]
    return lr, model, hist
