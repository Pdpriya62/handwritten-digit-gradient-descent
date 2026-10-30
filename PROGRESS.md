# PROGRESS

Last updated: 2026-10-30

## Status: Code, MNIST experiments and demo done; report/slides still to write

Repository: https://github.com/Pdpriya62/handwritten-digit-gradient-descent

## Checklist

### Setup
- [x] Project README drafted
- [x] Folder structure, `requirements.txt`, `.gitignore`
- [x] Create the GitHub repo and push

### Data
- [x] MNIST loader with local cache (`src/data.py`)
- [x] Normalise to 0-1, flatten to 784, train / validation / test split (50k / 10k / 10k)
- [x] First MNIST download (cached in `data/mnist.npz`)

### Implementation
- [x] Numerically stable Softmax and log-softmax
- [x] Cross-entropy loss
- [x] Gradients for W and b
- [x] Gradient check against finite differences (`src/gradcheck.py`)
- [x] Gradient Descent training loop and learning-rate sweep (`src/train.py`)
- [x] Predict, save and load weights

### Experiments (run on MNIST)
- [x] Baseline learning rate run (lr = 0.5)
- [x] Learning-rate sweep (0.01, 0.1, 0.5, 2, 8)
- [x] Loss curves (`results/loss_curves.png`)
- [x] Validation accuracy comparison (`results/accuracy_comparison.png`)

### Evaluation
- [x] Evaluation and plotting code (`src/evaluate.py`)
- [x] Test accuracy on MNIST
- [x] Confusion matrix on MNIST (`results/confusion_matrix.png`)
- [x] Misclassified samples on MNIST (in the notebook)

### Demo
- [x] Streamlit drawing app (`app.py`) with MNIST-style preprocessing (`src/preprocess.py`)
- [ ] Try the app with hand-drawn digits and note how well they are recognised

### Writing
- [ ] Gradient derivation written up
- [x] Observations in the notebook (learning-rate cells)
- [x] Results and discussion (notebook Conclusions cell)
- [ ] Final report / slides

### Optional future work
- [ ] Mini-batch SGD, L2 regularisation, momentum, neural network comparison

## Results log
Full-batch Gradient Descent, seed 0, same split for every run.

| Date | Experiment | LR | Epochs | Final train loss | Val acc | Notes |
|---|---|---|---|---|---|---|
| 2026-10-29 | Gradient check | - | - | - | - | Relative error 1.70e-10; untrained loss 2.3058 (ln 10 = 2.3026) |
| 2026-10-29 | LR sweep | 0.01 | 100 | 1.5622 | 0.7888 | Smooth but too slow |
| 2026-10-29 | LR sweep | 0.1 | 100 | 0.6180 | 0.8778 | Smooth |
| 2026-10-29 | LR sweep | 0.5 | 100 | 0.3980 | 0.9056 | Best; smooth |
| 2026-10-29 | LR sweep | 2 | 100 | 0.5149 | 0.8634 | Loss spikes, oscillates |
| 2026-10-29 | LR sweep | 8 | 100 | 1.6277 | 0.8457 | Large spikes (loss above 40) |
| 2026-10-29 | Test set | 0.5 | 100 | - | - | Test accuracy 0.9026, test loss 0.3720 |

Most common test confusions (true -> predicted): 4 -> 9 (49), 9 -> 4 (44), 2 -> 8 (44), 5 -> 3 (43), 7 -> 9 (40).

## Decisions
- Core project = Python modules plus a Jupyter notebook.
- Optional demo front end = Streamlit with a drawing canvas.
- Full-batch Gradient Descent, same seed and epochs for every learning rate.
- MNIST is downloaded as a plain `.npz` file. `fetch_openml` failed with a pandas parser error.
- Data is stored as `float64`, and `dW` is computed as `(X^T P)^T`. Together this cut one epoch from about 11.8 s to about 0.3 s.
- `st_canvas(..., return_image_data=True)` is required by streamlit-drawable-canvas 0.13.

## Blockers
- None.

## Next steps
1. Try `streamlit run app.py` with your own drawings and note the results.
2. Write up the gradient derivation (chain rule through Softmax and cross-entropy).
3. Make the final report or slides from the notebook results and the plots in `results/`.
