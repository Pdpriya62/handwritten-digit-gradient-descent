"""Run the whole pipeline: gradient check -> learning-rate sweep -> evaluation -> saved plots/weights.

    python run_all.py                       # MNIST, 100 epochs
    python run_all.py --epochs 200 --lrs 0.01 0.1 0.5 2 8
    python run_all.py --dataset digits      # tiny 8x8 data, only for a quick offline smoke test
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.data import ROOT, load_data
from src.evaluate import (evaluate, plot_accuracy_comparison, plot_confusion_matrix,
                          plot_loss_curves, plot_predictions, plot_weight_templates)
from src.gradcheck import gradient_check
from src.train import best_by_val_acc, lr_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "digits"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.1, 0.5, 2.0, 8.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()
    out = Path(args.out)

    data = load_data(args.dataset, seed=args.seed)
    print(f"train {len(data['y_train'])}  val {len(data['y_val'])}  test {len(data['y_test'])}")

    err = gradient_check(data["X_train"], data["y_train"])
    print(f"Gradient check, relative error: {err:.2e}  ({'OK' if err < 1e-6 else 'CHECK THE GRADIENTS'})")

    print(f"Training {len(args.lrs)} learning rates for {args.epochs} epochs each...")
    results = lr_sweep(data, args.lrs, epochs=args.epochs, seed=args.seed, verbose=True)

    best_lr, model, _ = best_by_val_acc(results)
    test = evaluate(model, data["X_test"], data["y_test"])

    print("\nlr        final train loss   final val acc")
    for lr, (_, h) in results.items():
        print(f"{lr:<9g} {h['train_loss'][-1]:<18.4f} {h['val_acc'][-1]:.4f}")
    print(f"\nBest learning rate by validation accuracy: {best_lr:g}")
    print(f"Test accuracy: {test['accuracy']:.4f}   test loss: {test['loss']:.4f}")

    plot_loss_curves(results, out / "loss_curves.png")
    plot_accuracy_comparison(results, out / "accuracy_comparison.png")
    plot_confusion_matrix(test["confusion_matrix"], out / "confusion_matrix.png")
    plot_predictions(model, data["X_test"], data["y_test"], data["image_shape"],
                     out / "misclassified.png", only_wrong=True)
    plot_weight_templates(model, data["image_shape"], out / "weight_templates.png")
    model.save(out / "weights.npz")

    summary = {
        "dataset": args.dataset, "epochs": args.epochs, "best_lr": best_lr,
        "test_accuracy": test["accuracy"], "test_loss": test["loss"],
        "gradient_check_rel_error": err,
        "per_lr": {str(lr): {"final_train_loss": h["train_loss"][-1],
                             "final_val_acc": h["val_acc"][-1]} for lr, (_, h) in results.items()},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved plots, weights.npz and metrics.json to {out}")


if __name__ == "__main__":
    main()
