"""Check the analytic gradients against central finite differences.

    numeric  dJ/dtheta ~ (J(theta + h) - J(theta - h)) / (2h)
"""
import numpy as np

from .model import SoftmaxClassifier


def gradient_check(X, y, n_samples=20, h=1e-5, seed=0):
    """Return ||numeric - analytic|| / (||numeric|| + ||analytic||) over all of W and b.

    A value around 1e-7 or smaller means the analytic gradients are correct.
    Using whole-vector norms avoids false alarms from entries whose gradient is ~0.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n_samples, len(y)), replace=False)
    Xs, ys = X[idx].astype(np.float64), y[idx]

    model = SoftmaxClassifier(Xs.shape[1], 10, seed=seed, init_scale=0.1)
    dW, db = model.gradients(Xs, ys)
    analytic = np.concatenate([dW.ravel(), db.ravel()])

    numeric = np.zeros_like(analytic)
    for param, offset in ((model.W, 0), (model.b, dW.size)):
        flat = param.reshape(-1)  # a view into the parameters, so edits change the model
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + h
            j_plus = model.loss(Xs, ys)
            flat[i] = old - h
            j_minus = model.loss(Xs, ys)
            flat[i] = old
            numeric[offset + i] = (j_plus - j_minus) / (2 * h)

    return float(np.linalg.norm(numeric - analytic) /
                 (np.linalg.norm(numeric) + np.linalg.norm(analytic) + 1e-12))


if __name__ == "__main__":
    from .data import load_data

    d = load_data("mnist")
    print("relative error:", gradient_check(d["X_train"], d["y_train"]))
