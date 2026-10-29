"""Dataset loading, normalisation, flattening and train/val/test splitting."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _load_mnist_raw(cache_dir=DATA_DIR):
    """Return uint8 images (70000, 784) and int labels. Downloads once, then caches."""
    cache = Path(cache_dir) / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    from sklearn.datasets import fetch_openml  # needs internet the first time only

    print("Downloading MNIST (one time, ~15 MB)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.uint8)
    y = y.astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_data(dataset="mnist", n_val=10000, seed=0, cache_dir=DATA_DIR):
    """Load a dataset and return a dict of float arrays scaled to [0, 1].

    dataset="mnist"  : 28x28 MNIST. Standard split: first 60000 = train pool,
                       last 10000 = test. The last `n_val` of the train pool become
                       the validation set.
    dataset="digits" : small 8x8 sklearn digits. Offline smoke-test only.
    """
    if dataset == "mnist":
        X, y = _load_mnist_raw(cache_dir)
        X = X.astype(np.float32) / 255.0
        X_pool, y_pool = X[:60000], y[:60000]
        X_test, y_test = X[60000:], y[60000:]
        X_train, y_train = X_pool[:-n_val], y_pool[:-n_val]
        X_val, y_val = X_pool[-n_val:], y_pool[-n_val:]
        shape = (28, 28)
    elif dataset == "digits":
        from sklearn.datasets import load_digits

        d = load_digits()
        X = (d.data / 16.0).astype(np.float32)
        y = d.target.astype(np.int64)
        order = np.random.default_rng(seed).permutation(len(y))
        X, y = X[order], y[order]
        a, b = int(0.6 * len(y)), int(0.8 * len(y))
        X_train, y_train = X[:a], y[:a]
        X_val, y_val = X[a:b], y[a:b]
        X_test, y_test = X[b:], y[b:]
        shape = (8, 8)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        image_shape=shape,
    )
