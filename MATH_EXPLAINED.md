# The Mathematics Behind the Project, Step by Step

This file explains every piece of mathematics used in the project, in the order the computer uses it. Each step has: the idea, the formula, a small worked calculation, and where it appears in the code.

**Contents**

1. [Images as vectors (linear algebra)](#1-images-as-vectors)
2. [Scores: a linear function](#2-scores-a-linear-function)
3. [Softmax: exponentials turn scores into probabilities](#3-softmax-exponentials-turn-scores-into-probabilities)
4. [Loss: logarithms and cross-entropy](#4-loss-logarithms-and-cross-entropy)
5. [Gradient: partial derivatives and the chain rule](#5-gradient-partial-derivatives-and-the-chain-rule)
6. [Matrix form: all 50,000 images at once](#6-matrix-form-all-50000-images-at-once)
7. [Gradient Descent: why stepping against the gradient works](#7-gradient-descent-why-stepping-against-the-gradient-works)
8. [The learning rate: curvature and the second derivative](#8-the-learning-rate-curvature-and-the-second-derivative)
9. [Convexity: why there is only one minimum](#9-convexity-why-there-is-only-one-minimum)
10. [Checking the derivative numerically](#10-checking-the-derivative-numerically)
11. [Measuring the result: accuracy and the confusion matrix](#11-measuring-the-result)
12. [Demo preprocessing: centre of mass](#12-demo-preprocessing-centre-of-mass)
13. [One complete worked example (all numbers)](#13-one-complete-worked-example)
14. [Summary table](#14-summary-table)

---

## 1. Images as vectors

**Math used:** vectors, scaling.

A grayscale image is a grid of pixel brightnesses. MNIST images are 28 x 28 = **784** pixels, each from 0 (black) to 255 (white). We read the grid row by row and write it as one long vector:

$$
x = (x_1, x_2, \dots, x_{784}) \in \mathbb{R}^{784}
$$

Then we **scale** every value into $[0, 1]$ by dividing by 255:

$$
x_j \leftarrow \frac{x_j}{255}
$$

Example: a pixel with brightness 204 becomes $204 / 255 = 0.8$.

Scaling keeps all inputs in the same small range, which keeps the numbers in the later steps (exponentials, gradients) well behaved.

*Code:* `src/data.py`, `X.astype(np.float64) / 255.0`

---

## 2. Scores: a linear function

**Math used:** linear functions, dot products, matrix-vector multiplication.

For each digit $k = 0, 1, \dots, 9$ the model has a **weight vector** $w_k \in \mathbb{R}^{784}$ (one weight per pixel) and a **bias** $b_k$. The score for digit $k$ is a dot product plus the bias:

$$
z_k = w_k \cdot x + b_k = \sum_{j=1}^{784} w_{kj}\, x_j + b_k
$$

Stacking the 10 weight vectors as rows of a matrix $W$ (size 10 x 784) gives all 10 scores at once:

$$
z = W x + b, \qquad W \in \mathbb{R}^{10 \times 784},\; b \in \mathbb{R}^{10},\; z \in \mathbb{R}^{10}
$$

**Meaning:** each weight says how much a bright pixel at that position is evidence for (positive weight) or against (negative weight) that digit. The model has $10 \times 784 + 10 = 7{,}850$ numbers to learn.

**Small example** (2 pixels, 1 digit): $w = (0.2, -0.1)$, $b = 0$, $x = (1, 0.5)$

$$
z = 0.2 \times 1 + (-0.1) \times 0.5 + 0 = 0.2 - 0.05 = 0.15
$$

*Code:* `src/model.py`, `scores()`: `X @ self.W.T + self.b`

---

## 3. Softmax: exponentials turn scores into probabilities

**Math used:** the exponential function, normalisation.

Scores can be any real number (negative, large, small). We want **probabilities**: each between 0 and 1, all adding up to 1. Softmax does this:

$$
p_k = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}
$$

**Why it works:**
- $e^{z} > 0$ for every real $z$, so every $p_k > 0$.
- Dividing by the total makes them add up to 1: $\sum_k p_k = \dfrac{\sum_k e^{z_k}}{\sum_j e^{z_j}} = 1$.
- $e^z$ is increasing, so a bigger score always gives a bigger probability. The order is kept.

**Example** with 3 classes and scores $z = (0.15,\ 0.15,\ 0.10)$:

| | class 0 | class 1 | class 2 |
|---|---|---|---|
| $z_k$ | 0.15 | 0.15 | 0.10 |
| $e^{z_k}$ | 1.1618 | 1.1618 | 1.1052 |
| $p_k = e^{z_k} / 3.4288$ | 0.3388 | 0.3388 | 0.3223 |

Sum of $e^{z_k}$ = 3.4288; sum of $p_k$ = 1.

The predicted digit is the class with the largest probability: $\hat{y} = \arg\max_k p_k$.

### 3.1 Stability trick (a property of exponentials)

$e^{1000}$ is too large for a computer (overflow). But subtracting the same constant $c$ from every score does not change softmax:

$$
\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_k}\, e^{-c}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

We choose $c = \max_j z_j$, so the largest exponent is $e^0 = 1$ and nothing overflows.

**Example:** $z = (1000, 1001, 1002)$. Subtract 1002: $(-2, -1, 0)$. Then $p = (0.0900,\ 0.2447,\ 0.6652)$, computed safely.

*Code:* `src/model.py`, `log_softmax()`: `shifted = z - z.max(axis=1, keepdims=True)`

---

## 4. Loss: logarithms and cross-entropy

**Math used:** the natural logarithm, averages.

We need **one number** that says how wrong the model is. For an image whose true digit is $y$, we look at the probability the model gave to the correct digit, $p_y$, and take

$$
\ell = -\ln p_y
$$

**Why the logarithm?**

| $p_y$ (probability of correct digit) | $-\ln p_y$ (loss) | Meaning |
|---|---|---|
| 1.0 | 0 | Perfect, no penalty |
| 0.9 | 0.105 | Small penalty |
| 0.5 | 0.693 | Unsure |
| 0.1 | 2.303 | Guessing |
| 0.01 | 4.605 | Confident and wrong, large penalty |

$-\ln p$ is 0 when $p = 1$ and grows without limit as $p \to 0$, so confident wrong answers are punished hardest. It also turns the softmax fraction into something easy to differentiate (section 5).

Over all $N$ training images, the **objective function** is the average loss (called **categorical cross-entropy**):

$$
J(W, b) = -\frac{1}{N} \sum_{i=1}^{N} \ln p_{i, y_i}
$$

Training means: **find $W$ and $b$ that make $J$ as small as possible.**

**Check before training:** with tiny random weights, all scores are about 0, so every $p_k \approx 1/10$ and

$$
J \approx -\ln \tfrac{1}{10} = \ln 10 \approx 2.3026
$$

Our untrained model measured **2.3058**, matching this prediction.

**Example:** in the table in section 3, if the true class is 2, then $\ell = -\ln 0.3223 = 1.1322$.

*Code:* `src/model.py`, `loss()`: `-logp[np.arange(len(y)), y].mean()`

---

## 5. Gradient: partial derivatives and the chain rule

**Math used:** partial derivatives, the chain rule, derivative of $\ln$ and $e^x$.

To lower $J$, we need to know how $J$ changes when each of the 7,850 parameters changes a little. That is the **partial derivative** $\partial J / \partial w_{kj}$. The vector of all partial derivatives is the **gradient** $\nabla J$.

$J$ is built in layers: parameters → scores $z$ → probabilities $p$ → loss $\ell$. So we use the **chain rule**:

$$
\frac{\partial \ell}{\partial w_{kj}} = \frac{\partial \ell}{\partial z_k} \cdot \frac{\partial z_k}{\partial w_{kj}}
$$

### Step 1: rewrite the loss using log rules

Using $\ln \dfrac{a}{b} = \ln a - \ln b$ and $\ln e^{z} = z$:

$$
\ell = -\ln \frac{e^{z_y}}{\sum_j e^{z_j}} = -z_y + \ln \sum_j e^{z_j}
$$

### Step 2: differentiate with respect to a score $z_k$

- First term: $\dfrac{\partial (-z_y)}{\partial z_k} = -1$ if $k = y$, otherwise $0$. We write this as $-\mathbf{1}[k = y]$ (the **indicator**: 1 if true, 0 if false).
- Second term: the chain rule with $\dfrac{d}{du} \ln u = \dfrac{1}{u}$ and $\dfrac{d}{dz} e^{z} = e^{z}$:

$$
\frac{\partial}{\partial z_k} \ln \sum_j e^{z_j} = \frac{1}{\sum_j e^{z_j}} \cdot e^{z_k} = p_k
$$

Adding both terms:

$$
\boxed{\frac{\partial \ell}{\partial z_k} = p_k - \mathbf{1}[k = y]}
$$

This is the key result: **predicted probability minus the truth.**

> **Alternative route (same answer):** the softmax derivative by the quotient rule is
> $\dfrac{\partial p_y}{\partial z_k} = p_y(\mathbf{1}[k = y] - p_k)$.
> Then $\dfrac{\partial \ell}{\partial z_k} = -\dfrac{1}{p_y} \cdot p_y(\mathbf{1}[k = y] - p_k) = p_k - \mathbf{1}[k = y]$.

### Step 3: differentiate the score with respect to the parameters

From $z_k = \sum_j w_{kj} x_j + b_k$:

$$
\frac{\partial z_k}{\partial w_{kj}} = x_j, \qquad \frac{\partial z_k}{\partial b_k} = 1
$$

### Step 4: multiply (chain rule)

$$
\frac{\partial \ell}{\partial w_{kj}} = (p_k - \mathbf{1}[k = y])\, x_j, \qquad \frac{\partial \ell}{\partial b_k} = p_k - \mathbf{1}[k = y]
$$

### Step 5: average over all images

The derivative of an average is the average of the derivatives:

$$
\frac{\partial J}{\partial w_{kj}} = \frac{1}{N} \sum_{i=1}^{N} (p_{ik} - \mathbf{1}[y_i = k])\, x_{ij},
\qquad
\frac{\partial J}{\partial b_k} = \frac{1}{N} \sum_{i=1}^{N} (p_{ik} - \mathbf{1}[y_i = k])
$$

**What it means:**
- If the model gives digit $k$ **too much** probability ($p_k$ larger than the truth), the gradient is positive, so descent will **lower** the weights on the pixels that are bright in this image.
- If it gives the **correct** digit too little probability, $p_y - 1 < 0$, so descent will **raise** those weights.
- Dark pixels ($x_j = 0$) get no change, because they gave no evidence.

**Example** (true class 2, $p = (0.3388, 0.3388, 0.3223)$, $x = (1, 0.5)$):

$$
p - \text{truth} = (0.3388,\ 0.3388,\ 0.3223 - 1) = (0.3388,\ 0.3388,\ -0.6777)
$$

The gradient for class 2's weights is $-0.6777 \times (1, 0.5) = (-0.6777,\ -0.3388)$. Negative means "increase these weights", which raises the score of the correct class.

---

## 6. Matrix form: all 50,000 images at once

**Math used:** matrix multiplication, transpose.

Writing a loop over 50,000 images is slow. Linear algebra does all of them in one product. Put the images as rows of $X$ ($N \times 784$), the probabilities as rows of $P$ ($N \times 10$), and the true labels as **one-hot** rows of $Y$ ($N \times 10$; a 1 in the column of the true digit, 0 elsewhere). Then:

$$
Z = X W^T + b, \qquad P = \text{softmax}(Z) \text{ (row by row)}
$$

$$
\nabla_W J = \frac{1}{N} (P - Y)^T X \qquad (10 \times N)(N \times 784) = 10 \times 784
$$

$$
\nabla_b J = \frac{1}{N} \sum_{i=1}^{N} (P - Y)_{i} \qquad (\text{column sums, length } 10)
$$

Entry $(k, j)$ of $(P - Y)^T X$ is $\sum_i (p_{ik} - y_{ik}) x_{ij}$, exactly the sum in step 5 above. The shapes always match the parameter they belong to: $\nabla_W J$ has the same shape as $W$.

Note: $(P - Y)^T X = \left(X^T (P - Y)\right)^T$ (transpose of a product, $(AB)^T = B^T A^T$). The code uses the second form because it runs much faster on this computer's linear algebra library. The result is identical.

*Code:* `src/model.py`, `loss_and_grads()`: `P[idx, y] -= 1.0` makes $P - Y$; `dW = (X.T @ P).T / N`; `db = P.sum(axis=0) / N`

---

## 7. Gradient Descent: why stepping against the gradient works

**Math used:** linear (first-order Taylor) approximation, directional derivative.

The gradient points in the direction where $J$ **increases fastest**. So we step in the **opposite** direction:

$$
W \leftarrow W - \alpha \nabla_W J, \qquad b \leftarrow b - \alpha \nabla_b J
$$

$\alpha > 0$ is the **learning rate** (step size). One update using all training images is one **epoch**. We do 100 epochs.

**Why it decreases the loss:** near the current point $\theta$, a function looks like its tangent plane (first-order Taylor expansion):

$$
J(\theta + \Delta) \approx J(\theta) + \nabla J \cdot \Delta
$$

Put in our step $\Delta = -\alpha \nabla J$:

$$
J(\theta - \alpha \nabla J) \approx J(\theta) - \alpha \, \|\nabla J\|^2
$$

Since $\alpha > 0$ and $\|\nabla J\|^2 \ge 0$, the loss goes **down** (as long as the step is small enough for the tangent-plane approximation to hold, see section 8). When $\nabla J = 0$ the steps stop: we are at a minimum.

**Example** (continuing, $\alpha = 0.5$, true class 2):

$$
w_2^{\text{new}} = (-0.1,\ 0.4) - 0.5 \times (-0.6777,\ -0.3388) = (0.2388,\ 0.5694)
$$

After updating all weights and biases, the probability of the correct class rises from **0.3223 to 0.5988** and the loss falls from **1.1322 to 0.5128** in one step. (Full numbers in section 13.)

*Code:* `src/train.py`, `model.W -= lr * dW` and `model.b -= lr * db`

---

## 8. The learning rate: curvature and the second derivative

**Math used:** second derivative, geometric sequences, eigenvalues.

### 8.1 A one-variable example

Take $f(w) = w^2$. Its minimum is at $w = 0$, $f'(w) = 2w$, and $f''(w) = 2$. Gradient Descent gives:

$$
w \leftarrow w - \alpha \cdot 2w = (1 - 2\alpha)\, w
$$

So after $t$ steps $w_t = (1 - 2\alpha)^t w_0$, a geometric sequence. It goes to 0 only if $|1 - 2\alpha| < 1$, which means $0 < \alpha < 1$. Starting from $w_0 = 1$:

| $\alpha$ | $1 - 2\alpha$ | $w_0, w_1, w_2, w_3, w_4$ | Behaviour |
|---|---|---|---|
| 0.1 | 0.8 | 1, 0.8, 0.64, 0.512, 0.41 | Converges slowly |
| 0.5 | 0 | 1, 0, 0, 0, 0 | Jumps straight to the minimum |
| 0.9 | -0.8 | 1, -0.8, 0.64, -0.512, 0.41 | **Oscillates** but converges |
| 1.1 | -1.2 | 1, -1.2, 1.44, -1.728, 2.07 | **Diverges**, overshoots more each time |

The limit is $\alpha < 1 = 2 / f''$. In general: **if the curvature (second derivative) is at most $L$, Gradient Descent is guaranteed to decrease the loss when $\alpha < 2/L$.** Steeper curvature means a smaller safe step.

### 8.2 The same idea for our model

With many variables, curvature is described by the **Hessian** (matrix of second derivatives), and the "largest curvature" is its largest **eigenvalue** $L$. For softmax cross-entropy it can be shown that

$$
L \le \tfrac{1}{2} \, \lambda_{\max}\!\left(\tfrac{1}{N} \tilde{X}^T \tilde{X}\right)
$$

where $\tilde{X}$ is $X$ with an extra column of 1s for the bias. The $\tfrac12$ is the largest eigenvalue of the softmax derivative matrix $\operatorname{diag}(p) - p p^T$.

For our training data, $\lambda_{\max} = 39.21$, so $L \le 19.60$ and

$$
\alpha < \frac{2}{L} \approx 0.102 \quad \text{is guaranteed safe.}
$$

### 8.3 What we observed

| $\alpha$ | Final training loss | Matches the 1-D picture |
|---|---|---|
| 0.01 | 1.5622 | Converging, but slowly (like $\alpha = 0.1$ above) |
| 0.1 | 0.6180 | Smooth, inside the guaranteed range |
| 0.5 | 0.3980 | Smooth and fastest. Above the worst-case bound, but the real curvature along the path is smaller |
| 2 | 0.5149 | Loss spikes up and down: overshooting (like $\alpha = 0.9$ / $1.1$) |
| 8 | 1.6277 | Loss jumps above 40: strong overshooting |

The bound $2/L$ is a **sufficient** condition (always safe), not a necessary one, which is why 0.5 still worked.

---

## 9. Convexity: why there is only one minimum

**Math used:** convex functions, second derivative test.

A function is **convex** if the line segment between any two points on its graph lies on or above the graph (a "bowl" shape). For one variable, $f'' \ge 0$ everywhere means convex. For convex functions, **every local minimum is a global minimum**, so Gradient Descent cannot get stuck in a "wrong" valley.

Our loss for one image is

$$
\ell = -z_y + \ln \sum_j e^{z_j}
$$

- $-z_y$ is linear in the parameters (convex).
- $\ln \sum_j e^{z_j}$ (log-sum-exp) is convex. One-variable check with two terms, $g(z) = \ln(e^z + 1)$: $g''(z) = \dfrac{e^z}{(e^z + 1)^2} > 0$.
- $z$ is a linear function of $W, b$, and a convex function of a linear function is convex.
- A sum (and average) of convex functions is convex.

So $J(W, b)$ is convex. The difficulty in training is **only** choosing a good step size (section 8), not avoiding bad minima.

---

## 10. Checking the derivative numerically

**Math used:** the definition of the derivative, central differences, vector norms.

How do we know the formula in section 5 is right? We compare it with the **definition** of the derivative, computed directly:

$$
\frac{\partial J}{\partial \theta} \approx \frac{J(\theta + h) - J(\theta - h)}{2h}, \qquad h = 10^{-5}
$$

This is the **central difference**. Its error shrinks like $h^2$ (from the Taylor expansion, the $h^1$ terms cancel), so it is much more accurate than $\frac{J(\theta + h) - J(\theta)}{h}$.

**Example** (the worked example, parameter $w_{2,0}$):
- numerical: $-0.67768367425$
- formula: $-0.67768367427$

We do this for every parameter, then compare the two gradient vectors with the **relative error**:

$$
\text{error} = \frac{\| g_{\text{numeric}} - g_{\text{formula}} \|}{\| g_{\text{numeric}} \| + \| g_{\text{formula}} \|}
$$

where $\|v\| = \sqrt{\sum v_i^2}$ is the length of a vector. Result for the real model: **1.70e-10**. Below about 1e-6 means the derivative is correct.

*Code:* `src/gradcheck.py`

---

## 11. Measuring the result

**Math used:** argmax, proportions, counting in a matrix.

- **Prediction:** $\hat{y}_i = \arg\max_k p_{ik}$ (the digit with the highest probability).
- **Accuracy:** $\dfrac{\text{number of } i \text{ with } \hat{y}_i = y_i}{N}$. Test accuracy = 9,026 / 10,000 = **0.9026**.
- **Confusion matrix:** a 10 x 10 table $C$ where $C_{tk}$ = number of images of true digit $t$ predicted as $k$. The diagonal holds correct answers, everything off the diagonal is a mistake. Largest off-diagonal entry: $C_{4,9} = 49$ (4s read as 9s).
- **Train / validation / test split:** the learning rate is chosen on validation data, and the test data is used only once, so the test accuracy is a fair estimate on new images.

*Code:* `src/evaluate.py`

---

## 12. Demo preprocessing: centre of mass

**Math used:** weighted averages (centre of mass), scaling ratios.

MNIST digits were prepared by fitting each digit into a 20 x 20 box and placing it so its **centre of mass** is at the centre of the 28 x 28 image. The demo does the same to a drawing:

1. Crop to the ink, then scale by $\dfrac{20}{\max(\text{height}, \text{width})}$ so the longer side is 20 pixels (the aspect ratio is kept).
2. Compute the centre of mass, treating brightness $I(r, c)$ as mass:

$$
\bar{r} = \frac{\sum_{r,c} r\, I(r,c)}{\sum_{r,c} I(r,c)}, \qquad \bar{c} = \frac{\sum_{r,c} c\, I(r,c)}{\sum_{r,c} I(r,c)}
$$

3. Shift the digit so $(\bar{r}, \bar{c})$ lands at the centre $(14, 14)$, then scale brightness to $[0, 1]$.

*Code:* `src/preprocess.py`

---

## 13. One complete worked example

A tiny model: **3 classes, 2 pixels**, one image, learning rate $\alpha = 0.5$. Every number below was computed exactly as the real code does it (rounded to 4 decimals).

**Given**

$$
x = (1,\ 0.5), \quad y = 2, \quad
W = \begin{pmatrix} 0.2 & -0.1 \\ 0.0 & 0.3 \\ -0.1 & 0.4 \end{pmatrix}, \quad
b = (0,\ 0,\ 0)
$$

**Step 1. Scores** $z = Wx + b$

$$
z_0 = 0.2(1) - 0.1(0.5) = 0.15, \quad z_1 = 0(1) + 0.3(0.5) = 0.15, \quad z_2 = -0.1(1) + 0.4(0.5) = 0.10
$$

**Step 2. Softmax**

$$
e^{z} = (1.1618,\ 1.1618,\ 1.1052), \quad \text{sum} = 3.4288, \quad p = (0.3388,\ 0.3388,\ 0.3223)
$$

**Step 3. Loss**

$$
\ell = -\ln p_2 = -\ln 0.3223 = 1.1322
$$

**Step 4. Error signal** $p - \text{one-hot}(y)$

$$
p - (0, 0, 1) = (0.3388,\ 0.3388,\ -0.6777)
$$

**Step 5. Gradients** $\nabla_W = (p - y)\, x^T$, $\nabla_b = p - y$

$$
\nabla_W = \begin{pmatrix} 0.3388 & 0.1694 \\ 0.3388 & 0.1694 \\ -0.6777 & -0.3388 \end{pmatrix}, \qquad
\nabla_b = (0.3388,\ 0.3388,\ -0.6777)
$$

**Step 6. Update** $W \leftarrow W - 0.5\, \nabla_W$, $b \leftarrow b - 0.5\, \nabla_b$

$$
W = \begin{pmatrix} 0.0306 & -0.1847 \\ -0.1694 & 0.2153 \\ 0.2388 & 0.5694 \end{pmatrix}, \qquad
b = (-0.1694,\ -0.1694,\ 0.3388)
$$

**Step 7. Check: forward pass again**

$$
z = (-0.2312,\ -0.2312,\ 0.8624), \quad p = (0.2006,\ 0.2006,\ 0.5988), \quad \ell = -\ln 0.5988 = 0.5128
$$

| | Before | After one step |
|---|---|---|
| Probability of correct class | 0.3223 | **0.5988** |
| Loss | 1.1322 | **0.5128** |

One Gradient Descent step moved the model towards the right answer. The real project repeats exactly this, with 784 pixels, 10 classes and 50,000 images, 100 times.

**Step 8. Gradient check on $w_{2,0}$:** numerical $-0.6777$, formula $-0.6777$ (they agree to 10 decimal places).

---

## 14. Summary table

| Step | Mathematics | Formula | Code |
|---|---|---|---|
| Input | Vectors, scaling | $x \in [0,1]^{784}$ | `src/data.py` |
| Scores | Linear function, matrix product | $z = Wx + b$ | `scores()` |
| Probabilities | Exponential, normalisation | $p_k = e^{z_k} / \sum_j e^{z_j}$ | `softmax()` |
| Stability | Exponent rules | subtract $\max z$ | `log_softmax()` |
| Objective | Logarithm, average | $J = -\frac1N \sum \ln p_{i,y_i}$ | `loss()` |
| Gradient | Partial derivatives, chain rule | $\nabla_W J = \frac1N (P - Y)^T X$ | `loss_and_grads()` |
| Optimisation | Taylor approximation | $\theta \leftarrow \theta - \alpha \nabla J$ | `train()` |
| Step size | Second derivative, eigenvalues | $\alpha < 2/L \approx 0.102$ | learning-rate sweep |
| One minimum | Convexity | $f'' \ge 0$ | - |
| Verification | Definition of derivative | $\frac{J(\theta+h) - J(\theta-h)}{2h}$ | `gradcheck.py` |
| Evaluation | Argmax, proportions | accuracy = correct / total | `evaluate.py` |
| Demo | Centre of mass | $\bar r = \sum r I / \sum I$ | `preprocess.py` |
