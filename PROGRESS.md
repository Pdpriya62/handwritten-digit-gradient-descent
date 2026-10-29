# PROGRESS

Last updated: 2026-10-29

## Status: Code written; waiting for the first full MNIST run

The code was tested end to end on a small offline dataset (8x8 sklearn digits) because MNIST could not be downloaded where it was built. **No MNIST results exist yet.** Run `python run_all.py` on your machine to produce them.

## Checklist

### Setup
- [x] Project README drafted
- [x] Folder structure, `requirements.txt`, `.gitignore`
- [ ] Create the GitHub repo and push

### Data
- [x] MNIST loader with local cache (`src/data.py`)
- [x] Normalise to 0-1, flatten to 784, train / validation / test split (50k / 10k / 10k)
- [ ] First MNIST download on your machine

### Implementation
- [x] Numerically stable Softmax and log-softmax
- [x] Cross-entropy loss
- [x] Gradients for W and b
- [x] Gradient check against finite differences (`src/gradcheck.py`)
- [x] Gradient Descent training loop and learning-rate sweep (`src/train.py`)
- [x] Predict, save and load weights

### Experiments (run on MNIST)
- [ ] Baseline learning rate run
- [ ] Learning-rate sweep (0.01, 0.1, 0.5, 2, 8)
- [ ] Loss curves
- [ ] Validation accuracy comparison

### Evaluation
- [x] Evaluation and plotting code (`src/evaluate.py`)
- [ ] Test accuracy on MNIST
- [ ] Confusion matrix on MNIST
- [ ] Misclassified samples on MNIST

### Demo
- [x] Streamlit drawing app (`app.py`) with MNIST-style preprocessing (`src/preprocess.py`)
- [ ] Try the app with real MNIST weights and note how well drawn digits are recognised

### Writing
- [ ] Gradient derivation written up
- [ ] Observations in the notebook (learning-rate cells)
- [ ] Results and discussion
- [ ] Final report / slides

### Optional future work
- [ ] Mini-batch SGD, L2 regularisation, momentum, neural network comparison

## Results log
| Date | Experiment | LR | Epochs | Val loss | Val acc | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Decisions
- Core project = Python modules plus a Jupyter notebook.
- Optional demo front end = Streamlit with a drawing canvas.
- Full-batch Gradient Descent, same seed and epochs for every learning rate.

## Blockers
- None.

## Next steps
1. `pip install -r requirements.txt`
2. `python run_all.py`, which downloads MNIST once, trains, and saves plots and `results/weights.npz`.
3. Open the notebook, run all, and write your observations.
4. `streamlit run app.py` for the demo.
