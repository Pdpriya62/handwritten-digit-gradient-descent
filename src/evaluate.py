"""Metrics and plots. Every plot function returns the figure and saves it if `path` is given."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def _finish(fig, path):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": float((pred == y).mean()),
        "loss": model.loss(X, y),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(10)),
    }


def plot_loss_curves(results, path=None):
    """results: {lr: (model, history)}. Training loss on a log-scaled y-axis."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr, (_, h) in results.items():
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=f"lr = {lr:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss J(W, b)")
    ax.set_yscale("log")
    ax.set_title("Training loss vs epoch for different learning rates")
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, path)


def plot_accuracy_comparison(results, path=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr, (_, h) in results.items():
        ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=f"lr = {lr:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy vs epoch for different learning rates")
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, path)


def plot_confusion_matrix(cm, path=None, title="Confusion matrix (test set)"):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, path)


def plot_predictions(model, X, y, image_shape, path=None, n=16, only_wrong=False):
    """Grid of images with predicted vs true labels (green = correct, red = wrong)."""
    pred = model.predict(X)
    idx = np.where(pred != y)[0] if only_wrong else np.arange(len(y))
    idx = idx[:n]
    cols = 8
    rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i].reshape(image_shape), cmap="gray")
        ax.set_title(f"pred {pred[i]} / true {y[i]}", fontsize=7,
                     color="green" if pred[i] == y[i] else "red")
    fig.suptitle("Misclassified test images" if only_wrong else "Sample predictions")
    return _finish(fig, path)


def plot_weight_templates(model, image_shape, path=None):
    """Each row of W reshaped as an image: what pattern each class 'looks for'."""
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for k, ax in enumerate(axes.ravel()):
        w = model.W[k].reshape(image_shape)
        m = np.abs(w).max()
        ax.imshow(w, cmap="seismic", vmin=-m, vmax=m)
        ax.set_title(f"class {k}")
        ax.axis("off")
    fig.suptitle("Learned weight vector for each digit")
    return _finish(fig, path)
