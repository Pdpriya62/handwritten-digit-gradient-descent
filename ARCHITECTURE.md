# ARCHITECTURE

## 1. Overview
A small, library-light ML pipeline. The core is plain Python and NumPy. A web front end is optional and only used for a demo.

```
                MNIST data
                    |
             [ data loading ]  -> normalise, flatten, split
                    |
   +----------------+-----------------+
   |                                  |
[ model.py ]                     [ train.py ]
 softmax, loss,        <-------   gradient descent loop,
 gradients, predict               LR sweep, logs
   |                                  |
   +----------------+-----------------+
                    |
             [ evaluate.py ]  -> accuracy, confusion matrix
                    |
        results/*.png  +  saved weights (.npz)
                    |
        (optional) app.py  -> Streamlit drawing demo
```

## 2. Modules
| File | Responsibility |
|---|---|
| `src/data.py` | Download MNIST once (`mnist.npz`), cache in `data/`, scale to 0-1, flatten, split train/val/test |
| `src/model.py` | `SoftmaxClassifier`: `forward`, `loss`, `gradients`, `predict` |
| `src/train.py` | Training loop, learning-rate sweep, loss/accuracy history |
| `src/evaluate.py` | Accuracy, confusion matrix, sample predictions, plots |
| `src/gradcheck.py` | Finite-difference check of analytic gradients |
| `src/preprocess.py` | Turns a canvas drawing into an MNIST-style 28x28 input |
| `run_all.py` | Runs the whole pipeline and writes `results/` (plots, `weights.npz`, `metrics.json`) |
| `notebooks/digit_classification.ipynb` | Runs experiments and shows plots |
| `app.py` (optional) | Streamlit demo that loads saved weights |

## 3. Data flow
1. `data.py` returns `X_train, y_train, X_val, y_val, X_test, y_test`.
2. `train.py` builds `SoftmaxClassifier(784, 10)` with a fixed seed.
3. Each epoch: forward pass, loss, gradients, update `W` and `b`, then record train loss, val loss and val accuracy.
4. The best run's weights are saved to `results/weights.npz`.
5. `evaluate.py` loads the weights and produces the test metrics and plots.

## 4. Key design decisions
- **NumPy only for the model**, so the calculus is visible.
- **Stable Softmax** (subtract the row max).
- **Fixed seed, split and epochs** across the learning-rate sweep for a fair comparison.
- **Weights stored as `.npz`**, so the demo does not retrain.
- **Scikit-learn only for metrics**, not for the model.
- **MNIST downloaded as a plain `.npz` file**, not with `fetch_openml`, which fails with some pandas versions.
- **`float64` data and weights.** Mixing `float32` data with `float64` weights made each matrix product about 10x slower.
- **`dW` computed as `(X^T P)^T`.** It equals `P^T X`, but avoids a very slow OpenBLAS path (about 11.8 s per epoch before both fixes, about 0.3 s after).

## 5. Front end decision
| Need | Choice |
|---|---|
| Coursework / report | Notebook + plots, with no UI |
| Live demo (draw a digit, get a prediction) | **Streamlit** with `streamlit-drawable-canvas` |
| Multi-user product | Not needed for this project |

### Optional demo architecture
```
Browser (Streamlit canvas)
        |  user draws a digit
        v
app.py: crop, scale to fit 20x20, centre by centre of mass in 28x28, scale 0-1, flatten
        |
        v
SoftmaxClassifier.predict(weights.npz)
        |
        v
Show predicted digit + probability bar chart
```
Note: canvas drawings differ from MNIST (centring, stroke thickness), so accuracy on drawn digits may be lower. Centre and resize the drawing before predicting.

## 6. Dependencies
`numpy`, `matplotlib`, `scikit-learn`, `jupyter`. For the demo: `streamlit`, `streamlit-drawable-canvas`, `pillow`.

## 7. Testing
- Gradient check: analytic vs numerical gradient, with relative error as small as possible.
- Loss on an untrained model should be near `ln(10)`.
- Overfit a tiny subset as a sanity check.
