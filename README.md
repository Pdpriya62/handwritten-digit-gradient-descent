# Handwritten Digit Classification Using Gradient Descent

## 📌 Project Overview

This project explores how **Gradient Descent, a fundamental optimization technique from Calculus, can be used to train a machine learning model** to recognize handwritten digits from 0 to 9.

The project implements a Softmax Regression classifier using NumPy and the MNIST handwritten digit dataset. Instead of relying entirely on machine learning libraries, the model's core mathematical operations—including the loss function, gradients, and parameter updates—are implemented from scratch.

The project also investigates how different learning rates affect convergence, training loss, and classification accuracy.

## 🎯 Objectives

- Understand the mathematical foundations of Gradient Descent.
- Derive partial derivatives of the categorical cross-entropy loss function.
- Implement a handwritten digit classifier using NumPy.
- Analyze the impact of learning rates on optimization.
- Evaluate model performance using accuracy and a confusion matrix.
- Connect Calculus concepts with a practical Artificial Intelligence and Machine Learning application.

## 🧠 Problem Statement

Recognizing handwritten digits is a common image classification problem in machine learning. Given an image of a handwritten digit, the model must predict which digit it represents, from 0 to 9.

Each MNIST image contains 28 × 28 grayscale pixels. These pixels are converted into a feature vector and passed to a trainable classification model.

The challenge is to find suitable model parameters that minimize prediction error. Gradient Descent solves this optimization problem by iteratively updating the model parameters in the direction opposite to the gradient of the loss function.

## 📊 Dataset

**MNIST Handwritten Digit Dataset**

- **Classes:** 10 digits (0–9)
- **Image dimensions:** 28 × 28 pixels
- **Features per image:** 784 pixel values
- **Pixel values:** 0–255, normalized to 0–1
- **Task:** Multiclass classification

Dataset source: [MNIST Database of Handwritten Digits](https://yann.lecun.com/exdb/mnist/)

The dataset will be divided into training, validation, and test sets to train the model, tune hyperparameters, and evaluate its final performance.

## 🧮 Mathematical Formulation

### 1. Model

Let \(x_i \in \mathbb{R}^{784}\) represent the flattened image of the \(i\)-th digit.

The model calculates a score for each of the ten classes:

\[
z_i = Wx_i+b
\]

where:

- \(W\in\mathbb{R}^{10\times784}\) is the weight matrix.
- \(b\in\mathbb{R}^{10}\) is the bias vector.
- \(z_i\in\mathbb{R}^{10}\) contains the class scores.

The Softmax function converts these scores into class probabilities:

\[
p_{ik}=\frac{e^{z_{ik}}}{\sum_{j=0}^{9}e^{z_{ij}}}
\]

The predicted digit is the class with the highest probability.

### 2. Objective Function

The model is trained by minimizing categorical cross-entropy:

\[
J(W,b)=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
\]

where:

- \(N\) is the number of training examples.
- \(y_i\) is the correct class of image \(i\).
- \(p_{i,y_i}\) is the probability assigned to the correct class.

The objective is to find parameters \(W\) and \(b\) that minimize \(J(W,b)\).

### 3. Gradient Descent

The gradient measures how the loss changes with respect to the model parameters.

For the weight associated with class \(k\):

\[
\frac{\partial J}{\partial w_k}
=\frac{1}{N}\sum_{i=1}^{N}
(p_{ik}-\mathbf{1}[y_i=k])x_i
\]

For the bias:

\[
\frac{\partial J}{\partial b_k}
=\frac{1}{N}\sum_{i=1}^{N}
(p_{ik}-\mathbf{1}[y_i=k])
\]

The parameters are updated iteratively:

\[
W_{t+1}=W_t-\alpha\nabla_WJ
\]

\[
b_{t+1}=b_t-\alpha\nabla_bJ
\]

Here, \(\alpha\) is the learning rate, which controls the size of each optimization step.

### 4. Learning Rate Analysis

The project will compare multiple learning rates to understand their effects on convergence.

- **Too small:** Training may converge slowly.
- **Appropriate:** The loss generally decreases steadily.
- **Too large:** Optimization may oscillate or become unstable.

Training loss and validation accuracy will be plotted against epochs to compare the behavior of different learning rates.

## ⚙️ Technologies Used

- **Python** — programming language
- **NumPy** — numerical computation and gradient calculations
- **Matplotlib** — plotting loss curves and sample predictions
- **Scikit-learn** — evaluation metrics and confusion matrix
- **Jupyter Notebook** — experimentation and visualization

The model's core training algorithm will be implemented using NumPy rather than a deep learning framework.

## 📁 Project Structure

```text
handwritten-digit-gradient-descent/
│
├── data/                    # Dataset files, if stored locally
├── notebooks/
│   └── digit_classification.ipynb
├── src/
│   ├── model.py              # Softmax classifier
│   ├── train.py              # Gradient Descent training
│   └── evaluate.py           # Model evaluation
├── results/
│   ├── loss_curves.png
│   ├── accuracy_comparison.png
│   └── confusion_matrix.png
├── requirements.txt
├── README.md
└── .gitignore
```

*This structure is a proposed organization for the project; files and plots will be added as they are implemented.*

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Pdpriya62/handwritten-digit-gradient-descent.git
cd handwritten-digit-gradient-descent
```


### 2. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
numpy
matplotlib
scikit-learn
jupyter
```

Then run:

```bash
pip install -r requirements.txt
```

### 4. Run the project

After implementing the training pipeline and preparing the dataset, launch the notebook:

```bash
jupyter notebook notebooks/digit_classification.ipynb
```

Follow the notebook to load the dataset, train the classifier, compare learning rates, and evaluate predictions.

## 📈 Results and Evaluation

The project will evaluate the model using:

- **Training loss:** Measures the objective value during training.
- **Validation loss:** Measures generalization to unseen validation examples.
- **Classification accuracy:** Percentage of correctly predicted digits.
- **Confusion matrix:** Shows which digits are most frequently confused.
- **Learning-rate comparison:** Illustrates differences in optimization behavior.
- **Sample predictions:** Displays images alongside their predicted and actual labels.

The final report will include the measured results after the experiments are completed. No accuracy or convergence result is assumed in advance.

## 🔬 Experiments

The following experiments will be conducted:

1. Train the classifier using a baseline learning rate.
2. Compare small, moderate, and large learning rates.
3. Plot training loss across epochs.
4. Compare validation accuracy for each learning rate.
5. Investigate whether the loss converges or oscillates.
6. Inspect correctly classified and misclassified digit images.

For a fair comparison, the experiments should use the same data split, parameter initialization where appropriate, and number of epochs.

## 📚 Calculus Concepts Demonstrated

This project connects theoretical Calculus with practical machine learning through:

- Partial derivatives of multivariable functions
- Gradients and directional change
- The chain rule used in differentiating the Softmax and cross-entropy composition
- Optimization of an objective function
- Iterative parameter updates
- Learning-rate effects on convergence

For Softmax Regression, the cross-entropy objective is convex in the model parameters under the standard formulation, although convergence speed depends on factors such as learning rate and data conditioning.

## 🌍 Real-World Applications

The concepts demonstrated in this project are relevant to:

- Digit recognition in postal codes and addresses
- Automated processing of handwritten forms
- Bank cheque and document processing
- Optical character recognition (OCR)
- Understanding optimization in larger neural networks

The implemented Softmax Regression model is a baseline classifier rather than a full deep neural network, but the Gradient Descent principles also apply to neural network training.

## 🔮 Future Improvements

- Implement mini-batch and stochastic Gradient Descent.
- Add L2 regularization.
- Compare Softmax Regression with a neural network.
- Experiment with momentum-based optimization.
- Build a simple interface where users can draw a digit and receive a prediction.
- Study the effect of feature scaling and parameter initialization.

## 👨‍💻 Author

**Priyadarshini Prasad** (Section D, AIML), RV University

Second-Year Computer Science and Engineering — Artificial Intelligence and Machine Learning

## 📄 License

This project may be released under the MIT License. Add a `LICENSE` file if you choose to publish it under that license.

---

**Key takeaway:** This project demonstrates how Calculus-based optimization enables a machine learning model to learn from data, while providing an opportunity to study gradients, objective functions, and convergence experimentally.

## ▶️ Quick Start (Demo)

```bash
pip install -r requirements.txt
python run_all.py          # downloads MNIST once, trains, saves plots + results/weights.npz
jupyter notebook notebooks/digit_classification.ipynb   # step-by-step experiments
streamlit run app.py       # draw a digit and get a prediction
```

Other files: `run_all.py` (whole pipeline), `app.py` (Streamlit demo), `src/gradcheck.py` (numerical gradient check), `src/preprocess.py` (canvas drawing to 28x28 input), `PROJECT.md`, `PROGRESS.md`, `ARCHITECTURE.md`.
