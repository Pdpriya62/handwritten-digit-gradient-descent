"""Softmax Regression implemented from scratch with NumPy.

Math (N examples, K = 10 classes, d = 784 features):
    z = X W^T + b                      scores, shape (N, K)
    p = softmax(z)                     probabilities
    J = -(1/N) * sum_i log p[i, y_i]   categorical cross-entropy
    dJ/dW = (1/N) * (P - Y)^T X        Y is the one-hot label matrix
    dJ/db = (1/N) * sum_i (P - Y)[i]
"""
from pathlib import Path

import numpy as np


def log_softmax(z):
    """Numerically stable log(softmax(z)) along axis 1."""
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(z):
    """Numerically stable softmax along axis 1 (subtract the row max first)."""
    return np.exp(log_softmax(z))


class SoftmaxClassifier:
    def __init__(self, n_features=784, n_classes=10, seed=0, init_scale=0.01):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, init_scale, size=(n_classes, n_features))
        self.b = np.zeros(n_classes)

    # ---- forward pass -------------------------------------------------
    def scores(self, X):
        return X @ self.W.T + self.b

    def predict_proba(self, X):
        return softmax(self.scores(X))

    def predict(self, X):
        return self.scores(X).argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    # ---- loss and gradients -------------------------------------------
    def loss(self, X, y):
        """Mean categorical cross-entropy J(W, b)."""
        logp = log_softmax(self.scores(X))
        return float(-logp[np.arange(len(y)), y].mean())

    def loss_and_grads(self, X, y):
        """Return (J, dJ/dW, dJ/db) using one forward pass."""
        N = X.shape[0]
        logp = log_softmax(self.scores(X))
        idx = np.arange(N)
        J = float(-logp[idx, y].mean())
        P = np.exp(logp)
        P[idx, y] -= 1.0              # P - Y, without building the one-hot matrix
        dW = (X.T @ P).T / N          # same as P^T X; this order avoids a very slow BLAS path
        db = P.sum(axis=0) / N
        return J, dW, db

    def gradients(self, X, y):
        _, dW, db = self.loss_and_grads(X, y)
        return dW, db

    # ---- persistence --------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(n_features=data["W"].shape[1], n_classes=data["W"].shape[0])
        model.W, model.b = data["W"], data["b"]
        return model
