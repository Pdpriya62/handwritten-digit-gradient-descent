# PROJECT: Handwritten Digit Classification Using Gradient Descent

**Author:** Priyadarshini Prasad (Section D, AIML), RV University
**Course context:** Calculus, applied to machine learning
**Status:** Planning / early implementation (see `PROGRESS.md`)

## 1. Summary
Train a Softmax Regression classifier on MNIST to recognise digits 0-9. The gradient, loss and parameter updates are implemented from scratch in NumPy. The project studies how the learning rate affects convergence and accuracy.

## 2. Goals
- Derive the gradients of categorical cross-entropy by hand (chain rule through Softmax).
- Implement the model, loss, gradients and training loop in NumPy only.
- Compare several learning rates on the same data split and epoch budget.
- Evaluate with accuracy, validation loss and a confusion matrix.
- Present the link between Calculus concepts and a real ML task.

## 3. Scope
**In scope:** Softmax Regression, batch Gradient Descent, learning-rate study, evaluation plots, written report.
**Out of scope (future work):** Mini-batch/SGD, L2 regularisation, momentum, neural network comparison, drawing interface.

## 4. Dataset
MNIST: 28 x 28 grayscale images, 784 features, 10 classes. Pixels are scaled from 0-255 to 0-1. The data is split into train, validation and test sets.

## 5. Method (short)
- Scores: `z = Wx + b`
- Probabilities: Softmax
- Loss: `J = -(1/N) * sum(log p[i, y_i])`
- Gradients: `dW = (1/N) * (P - Y)^T X`, `db = mean(P - Y)`
- Update: `W <- W - alpha * dW`, `b <- b - alpha * db`

## 6. Experiments
1. Baseline learning rate.
2. Small / moderate / large learning rates (same split, init, epochs).
3. Loss curves per learning rate.
4. Validation accuracy per learning rate.
5. Check convergence vs oscillation.
6. Inspect correct and misclassified samples.

## 7. Deliverables
- `src/` code (model, train, evaluate)
- `notebooks/digit_classification.ipynb`
- `results/` plots (loss curves, accuracy comparison, confusion matrix)
- Written report or slides with the derivation and findings
- Optional: small demo app (see `ARCHITECTURE.md`)

## 8. Success criteria
- Gradients verified against numerical (finite-difference) gradients.
- Loss decreases for a sensible learning rate; unstable behaviour shown for a too-large one.
- Results reported exactly as measured, with no assumed accuracy figures.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Numerical overflow in Softmax | Subtract the row max before `exp` |
| Slow full-batch training | Vectorise with NumPy; fewer epochs for the sweep |
| Unfair LR comparison | Fix split, init seed and epochs |
| Dataset download issues | Cache MNIST locally in `data/` |
