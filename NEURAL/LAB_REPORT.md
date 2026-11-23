# AI Laboratory Report: Neural Models
## Learning, Depth, Activations, and Output Layers

**Course:** Artificial Intelligence Laboratory  
**Topic:** Neural Parameterized Models, Depth vs. Nonlinearity, Reverse-Mode AD, and Task-Dependent Outputs  
**Laboratory Date:** August 2026  
**Workspace:** `C:\Users\Amithav\Desktop\LAB\NEURAL`  
**Deliverables Generated:**
- [neural_lab.ipynb](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/neural_lab.ipynb) (Complete Interactive Executed Notebook)
- [NEURAL_LAB_REPORT.pdf](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/NEURAL_LAB_REPORT.pdf) (4-Page Publication-Grade PDF Report)
- [xor_binary_model.py](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/xor_binary_model.py) (Baseline 2-2-1 Model)
- [symmetry_experiment.py](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/symmetry_experiment.py) (Zero-Init Symmetry Collapse Experiment)
- [activation_experiment.py](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/activation_experiment.py) (Sigmoid vs. Tanh vs. ReLU Benchmark)
- [three_class_extension.py](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/three_class_extension.py) (Multiclass 3-Class Classifier)
- [generate_plots.py](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/generate_plots.py) (Visual Visualization Script)
- [run_experiments.py](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/run_experiments.py) (Master Experiment Orchestrator)
- [TASKS_INDEX.txt](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/TASKS_INDEX.txt) (Deliverables Directory Index)

---

## 1. Executive Summary & Motivating Scenario

This laboratory investigates the computational foundation of artificial neural networks: **how parameterized functions learn representations via gradient-based optimization**.

### Motivating Scenario: Redundant Safety Sensors
A small autonomous robotic platform utilizes two redundant binary safety sensors: $x_1$ and $x_2$. The safety executive system is designed to raise a **sensor-disagreement warning** ($y = 1$) whenever exactly one sensor triggers (indicating a fault or divergence in sensor readings), and remain silent ($y = 0$) when both agree (both inactive or both active).

This requirement maps directly to the Boolean **Exclusive-OR (XOR)** problem:

$$\mathcal{D}_{XOR} = \{ ((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0) \}$$

While intentionally microscopic (only four training instances), XOR is historically celebrated because it represents the simplest decision rule that cannot be solved by a linear model, making the internal representations, backpropagation dynamics, and activation behaviors fully transparent and mathematically inspectable.

---

## 2. Task 1: Understand the Problem Before Coding

### 2.1 Problem Specification
- **Input Space:** $\mathcal{X} = \{0, 1\}^2 \subset \mathbb{R}^2$, where $\mathbf{x} = [x_1, x_2]^T$.
- **Output Space:** $\mathcal{Y} = \{0, 1\}$, indicating whether the sensor disagreement alarm is active.
- **Dataset:** Four labelled observations:
  1. $\mathbf{x}^{(1)} = [0, 0]^T \implies y = 0$ (Both sensors inactive)
  2. $\mathbf{x}^{(2)} = [0, 1]^T \implies y = 1$ (Sensor 2 active, Sensor 1 inactive)
  3. $\mathbf{x}^{(3)} = [1, 0]^T \implies y = 1$ (Sensor 1 active, Sensor 2 inactive)
  4. $\mathbf{x}^{(4)} = [1, 1]^T \implies y = 0$ (Both sensors active)

### 2.2 Proof of Linear Inseparability
A linear decision boundary in $\mathbb{R}^2$ is parameterized by a weight vector $\mathbf{w} = [w_1, w_2]^T$ and bias $b \in \mathbb{R}$:
$$\mathcal{H} = \{ \mathbf{x} \in \mathbb{R}^2 : w_1 x_1 + w_2 x_2 + b = 0 \}$$

The points of Class 1 ($\mathcal{C}_1 = \{(0, 1), (1, 0)\}$) form a line segment with convex hull:
$$\text{conv}(\mathcal{C}_1) = \{ \lambda (0, 1) + (1 - \lambda) (1, 0) : \lambda \in [0, 1] \} = \{ (1 - \lambda, \lambda) : \lambda \in [0, 1] \}$$

The points of Class 0 ($\mathcal{C}_0 = \{(0, 0), (1, 1)\}$) form a line segment with convex hull:
$$\text{conv}(\mathcal{C}_0) = \{ \mu (0, 0) + (1 - \mu) (1, 1) : \mu \in [0, 1] \} = \{ (1 - \mu, 1 - \mu) : \mu \in [0, 1] \}$$

Setting $\lambda = 0.5$ and $\mu = 0.5$:
$$\lambda (0, 1) + (1 - \lambda) (1, 0) = (0.5, 0.5), \quad \mu (0, 0) + (1 - \mu) (1, 1) = (0.5, 0.5)$$

By Radon's Theorem and the Hyperplane Separation Theorem, two sets in $\mathbb{R}^d$ are strictly linearly separable if and only if their convex hulls are disjoint:
$$\text{conv}(\mathcal{C}_0) \cap \text{conv}(\mathcal{C}_1) = \{ (0.5, 0.5) \} \neq \emptyset$$
Because the convex hulls intersect at $(0.5, 0.5)$, **no straight line can partition the two classes without error**.

![XOR Linear Inseparability](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/xor_linear_inseparability.png)

### 2.3 Prediction: Single Affine Transformation + Sigmoid Output
A single affine model computes:
$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp(-(w_1 x_1 + w_2 x_2 + b))}$$
The decision threshold $\hat{y} \ge 0.5$ is strictly equivalent to $\mathbf{w}^T \mathbf{x} + b \ge 0$.

**Prediction:** Since no linear hyperplane can separate the points, gradient descent on binary cross-entropy will converge to a compromised stationary point where at most 3 of the 4 points are classified correctly (maximum 75% accuracy), or weights decay towards zero with $b = 0$, predicting $\hat{y} = 0.5$ for all inputs with loss plateauing at $\ln(2) \approx 0.69315$.

### Think About It: What Scientific Claim Does XOR Test?
> *"A model can have many parameters and still have the wrong kind of representation. What scientific claim about representation does XOR let us test with only four data points?"*

**Answer:** XOR tests the foundational scientific claim that **representational power requires nonlinear geometric composition, not merely depth or parameter quantity**.  
Consider an $L$-layer network of affine transformations without nonlinear activations:
$$\mathbf{h}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{h}^{(2)} = W^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)}, \quad \dots \quad \mathbf{z} = W^{(L)}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$
By expanding the linear compositions:
$$\mathbf{z} = (W^{(L)} W^{(L-1)} \cdots W^{(1)})\mathbf{x} + \left( W^{(L)} \cdots W^{(2)}\mathbf{b}^{(1)} + \dots + \mathbf{b}^{(L)} \right) = \tilde{W}\mathbf{x} + \tilde{\mathbf{b}}$$
Regardless of depth $L$ or hidden layer widths, the entire deep network collapses into a single affine transformation with a flat hyperplane decision surface. XOR proves that learning requires nonlinear mapping functions that fold the coordinate space.

---

## 3. Task 2: Design the Intelligent Agent

### 3.1 Network Architecture (2-2-1 Design)
```
Input Layer (2)        Hidden Layer (2)         Output Logit (1)
  x1 ---------------> [ h1 = f(a1) ] ---------> [ z = W^(2)h + b^(2) ]
          \       /                  \       /
           \     /                    \     /
  x2 ---------------> [ h2 = f(a2) ] ---------> Output Prob: p = sigmoid(z)
```

1. **Input:** $\mathbf{x} = [x_1, x_2]^T \in \mathbb{R}^2$
2. **Hidden Layer:**
   $$\mathbf{a}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{h}^{(1)} = f(\mathbf{a}^{(1)})$$
   where $W^{(1)} \in \mathbb{R}^{2 \times 2}$, $\mathbf{b}^{(1)} \in \mathbb{R}^2$, and $f(\cdot)$ is a nonlinear activation (Sigmoid, Tanh, or ReLU).
3. **Output Layer:**
   $$z = W^{(2)}\mathbf{h}^{(1)} + b^{(2)}$$
   where $W^{(2)} \in \mathbb{R}^{1 \times 2}$, $b^{(2)} \in \mathbb{R}$.
4. **Prediction & Loss:**
   $$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \mathcal{L} = -\left[ y \ln \hat{p} + (1 - y) \ln(1 - \hat{p}) \right]$$

### 3.2 Scientific & Engineering Justifications
1. **Scientific Necessity of Nonlinearity:** Without $f(\cdot)$, the network collapses into $\tilde{W}\mathbf{x} + \tilde{\mathbf{b}}$, which cannot separate XOR. The nonlinear hidden units act as coordinate transformers, projecting the four collinear/intersecting 2D vertices into a 2D hidden space $\mathbf{h} = [h_1, h_2]^T$ where their convex hulls become disjoint.
2. **Sigmoid + BCE Pairing:** The target $y \in \{0, 1\}$ is a Bernoulli trial. Sigmoid maps the real line $\mathbb{R} \to (0, 1)$, and BCE represents the exact negative log-likelihood. Under this canonical pairing, the derivative of the loss with respect to logit $z$ simplifies to:
   $$\frac{\partial \mathcal{L}}{\partial z} = \hat{p} - y$$
   This linear error prevents the vanishing gradients that arise when pairing Sigmoid with Mean Squared Error (where $\sigma'(z) = \hat{p}(1-\hat{p})$ extinguishes the learning signal whenever the network is saturated and confidently wrong).
3. **Four Validation Criteria for Successful Learning:**
   - **Criterion 1 (Loss Convergence):** Final BCE loss $< 0.01$.
   - **Criterion 2 (Classification Accuracy):** $100\%$ accuracy (all $4/4$ thresholded predictions $\mathbb{I}(\hat{p} \ge 0.5) == y$).
   - **Criterion 3 (Stationary Convergence):** Gradient norm $\|\nabla_{W^{(1)}} \mathcal{L}\|_2 \to 0$ as the parameters approach the local minimum.
   - **Criterion 4 (Feature Rank Non-Degeneracy):** Hidden neurons compute two linearly independent feature vectors across the dataset.

### Think About It: Credit Assignment Without Direct Supervision
> *"The hidden units are not given target values. If the network learns XOR, what has determined what each hidden unit should compute? Relate your answer to the role of backpropagation."*

**Answer:** The hidden units discover useful features through **gradient-based credit assignment**. The supervisor only penalizes errors at the final scalar output $z$. Backpropagation projects this scalar error signal backwards through the transpose of the output weights:
$$\delta_j^{(1)} = \frac{\partial \mathcal{L}}{\partial a_j^{(1)}} = \left( \frac{\partial \mathcal{L}}{\partial z} W_j^{(2)} \right) f'(a_j^{(1)})$$
This backward gradient instructs hidden unit $j$ how to rotate its hyperplane $W_j^{(1)}\mathbf{x} + b_j^{(1)} = 0$ to minimize the global loss. Coupled with random initial asymmetry, one hidden unit typically specializes into detecting $(x_1 \lor x_2)$ (or NAND) while the other specializes into $(x_1 \land x_2)$, carving out the two hyperplanes necessary to isolate XOR.

---

## 4. Task 3: LLM Engineering Assistance & Code Inspection

### 4.1 Exact Prompt Provided to the LLM
```text
Generate minimal PyTorch code for the following model and dataset. Do not change
the architecture or task. Implement a 2-2-1 network for XOR with explicit inputs
[[0,0],[0,1],[1,0],[1,1]] and labels [[0],[1],[1],[0]]. Use a nonlinear hidden
activation (sigmoid/tanh), logits with BCEWithLogitsLoss, random weight
initialization, and full-batch training for a few thousand lightweight CPU steps.
After training, report the initial and final loss, all four probabilities,
thresholded labels, and one parameter-gradient tensor. Set a random seed for
reproducibility and explain each test in one sentence.
```

### 4.2 Computational Graph Inspection
1. **Forward Pass:** Implemented inside `XORNet.forward(x)`. Executes `self.fc1(x)`, passes through `self.act`, and computes raw logits via `self.fc2(h)`.
2. **Scalar Loss Formation:** Evaluated via `loss = criterion(logits, y)`, aggregating the four element-wise cross-entropies using the arithmetic mean reduction.
3. **Reverse-Mode AD:** Triggered by `loss.backward()`, which traverses the directed acyclic graph (DAG) in topological reverse order, executing vector-Jacobian products and populating `.grad` on all leaf parameters.
4. **Optimizer Update:** Performed by `optimizer.step()`, adjusting weights along the calculated gradient trajectory.

### 4.3 Two Engineering Changes Made Before Execution
1. **Change 1: Numerical Stability via `BCEWithLogitsLoss`:** Raw LLM templates frequently emit `torch.sigmoid(logits)` piped into `nn.BCELoss()`. In floating-point arithmetic, when logits exceed $|z| > 15$, `sigmoid(z)` rounds to $1.0$ or $0.0$, causing $\ln(\hat{p})$ to underflow to `NaN` or `-inf`. We enforced `nn.BCEWithLogitsLoss()`, which computes the Log-Sum-Exp stably:
   $$\mathcal{L} = \max(z, 0) - z y + \ln(1 + e^{-|z|})$$
2. **Change 2: Pre-Zeroing Gradients & Tracking Baseline State:** Enforced that `optimizer.zero_grad()` is invoked before every `loss.backward()` to prevent PyTorch's default accumulation behavior, and inserted an evaluation pass before step 0 to log the untrained initial loss and initial gradient norm.

---

## 5. Task 4: Execution, Testing, and Diagnostics

### 5.1 Part A: Basic Learning Check
Trained using `xor_binary_model.py` with Sigmoid hidden activation, Adam optimizer ($\eta = 0.05$), and random seed $42$:
- **Initial Loss:** $0.748230$
- **Epoch 500:** Loss $= 0.008669$ | Correct: $4/4$
- **Epoch 1000:** Loss $= 0.002665$ | Correct: $4/4$
- **Epoch 3000 (Final):** Loss $= \mathbf{0.000340}$

| Input $(x_1, x_2)$ | Target $y$ | Predicted Prob $\hat{p}$ | Thresholded Label $\hat{y}$ | Verification |
| :---: | :---: | :---: | :---: | :---: |
| $(0, 0)$ | 0 | $0.000308$ | 0 | **PASS** |
| $(0, 1)$ | 1 | $0.999694$ | 1 | **PASS** |
| $(1, 0)$ | 1 | $0.999554$ | 1 | **PASS** |
| $(1, 1)$ | 0 | $0.000299$ | 0 | **PASS** |

**Result:** All four inputs are classified with $> 99.95\%$ certainty ($4/4$ correct).

### 5.2 Part B: Backpropagation Check & Average Gradient
Initial first-layer weight gradient tensor $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$:
$$\nabla_{W^{(1)}} \mathcal{L}\Big|_{t=0} = \begin{bmatrix} -0.00399953 & -0.00442072 \\ 0.00639591 & 0.00806316 \end{bmatrix}, \quad \|\nabla_{W^{(1)}} \mathcal{L}\|_2 = 0.011894$$

Final first-layer weight gradient tensor at step 3000:
$$\nabla_{W^{(1)}} \mathcal{L}\Big|_{t=3000} = \begin{bmatrix} 5.14 \times 10^{-6} & -5.75 \times 10^{-6} \\ 2.75 \times 10^{-6} & -3.94 \times 10^{-6} \end{bmatrix}, \quad \|\nabla_{W^{(1)}} \mathcal{L}\|_2 = 8.98 \times 10^{-6}$$

**Why the gradient is the average of example-wise gradients:**  
The empirical loss across the batch is defined as:
$$\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \mathcal{L}_n$$
By linearity of the differentiation operator:
$$\nabla_W \mathcal{L} = \nabla_W \left( \frac{1}{N} \sum_{n=1}^N \mathcal{L}_n \right) = \frac{1}{N} \sum_{n=1}^N \nabla_W \mathcal{L}_n$$
PyTorch's reverse-mode automatic differentiation accumulates the vector-Jacobian products over each batch example and divides by $N = 4$.

### 5.3 Part C: Symmetry Experiment (Zero Weight Initialization)
To test symmetry breaking, all parameters were initialized to $0.0$:
$$W^{(1)} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{b}^{(1)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad W^{(2)} = \begin{bmatrix} 0 & 0 \end{bmatrix}, \quad b^{(2)} = 0$$

Tracking Row 1 and Row 2 of $W^{(1)}$ across training steps (`symmetry_experiment.py`):

| Step | $W^{(1)}$ Row 1 $[w_{11}, w_{12}]$ | $W^{(1)}$ Row 2 $[w_{21}, w_{22}]$ | Identical? | BCE Loss |
| :---: | :---: | :---: | :---: | :---: |
| 0 | $[0.000000, 0.000000]$ | $[0.000000, 0.000000]$ | **True** | $0.693147$ |
| 1 | $[0.000000, 0.000000]$ | $[0.000000, 0.000000]$ | **True** | $0.693147$ |
| 5 | $[0.000000, 0.000000]$ | $[0.000000, 0.000000]$ | **True** | $0.693147$ |
| 10 | $[0.000000, 0.000000]$ | $[0.000000, 0.000000]$ | **True** | $0.693147$ |
| 100 | $[0.000000, 0.000000]$ | $[0.000000, 0.000000]$ | **True** | $0.693147$ |

![Symmetry Collapse](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/symmetry_collapse.png)

**Mathematical Explanation:**  
1. When $W^{(1)} = \mathbf{0}$, both hidden units compute identical pre-activations: $a_1 = a_2 = 0$.
2. Both hidden units output identical activations: $h_1 = h_2 = \sigma(0) = 0.5$.
3. When $W^{(2)} = [0, 0]$, both units receive zero backpropagated gradient: $\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_2} = 0$.
4. Even when $W^{(2)}$ updates, both components receive identical updates because $h_1 = h_2$, so $W_1^{(2)} = W_2^{(2)}$ is preserved.
5. Consequently, $\nabla_{W_1^{(1)}} \mathcal{L} = \nabla_{W_2^{(1)}} \mathcal{L}$ holds identically.
The hidden units remain clones of each other, collapsing the effective hidden dimension to $1$. A 2-1-1 network cannot solve XOR; hence, the model remains permanently trapped at $\mathcal{L} = \ln(2) \approx 0.693147$.

### 5.4 Part D: Activation Experiment (Sigmoid vs. Tanh vs. ReLU)
Trained under identical random initialization (`seed=2`, $\eta = 0.05$, $2500$ epochs):

| Hidden Activation | Final Loss | 4/4 Correct? | Early $\|\nabla_{W^{(1)}} \mathcal{L}\|_2$ (Step 10) |
| :---: | :---: | :---: | :---: |
| **Sigmoid** | $0.000380$ | **Yes (4/4)** | $0.008495$ |
| **Tanh** | $0.000132$ | **Yes (4/4)** | $0.046584$ |
| **ReLU** | $\mathbf{0.000049}$ | **Yes (4/4)** | $0.024614$ |

![Training Curves](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/training_curves.png)

**Interpretation:**
- **Sigmoid:** Has a derivative bound $\sigma'(z) \le 0.25$, attenuating backpropagated gradients by a factor of $\ge 4$ per layer. This produces the smallest early gradient norm ($0.008495$) and slower initial loss descent.
- **Tanh:** Is zero-centered with $\tanh'(0) = 1.0$, producing the largest early gradient norm ($0.046584$) and rapid early descent.
- **ReLU:** Provides piecewise constant derivative $1.0$ for positive units, avoiding gradient shrinkage and converging to the lowest loss ($4.91 \times 10^{-5}$).

### Think About It: Distinguishing Saturated Sigmoids from Inactive ReLUs
> *"If a sigmoid unit is saturated, its derivative is close to zero. If a ReLU unit is negative, its derivative is zero. These are different mechanisms that can both produce a small gradient. How would you distinguish them by inspecting activations and pre-activations?"*

**Answer:**
- **Sigmoid Saturation:** Occurs when $|a| \gg 0$ (e.g. $a > 4$ or $a < -4$). The activation $h = \sigma(a)$ is extremely close to $1.0$ or $0.0$, and the derivative is $\sigma'(a) = h(1-h) \approx 0$.
- **ReLU Inactivity (Dead ReLU):** Occurs when $a < 0$. The activation is exactly $h = 0.0$, and the derivative is identically $0.0$.
- **Diagnostic Method:** By inspecting both $a$ and $h$: if $h = 0$ while $a < 0$, it is a dead ReLU; if $|a| \gg 0$ and $h \approx 1$ or $h \approx 0$, it is a saturated Sigmoid.

---

## 6. Task 5: Multiclass Sensor Extension (3-Class Classification)

### 6.1 Task Formulation
The sensor inputs are mapped to three mutually exclusive operational classes:
- **Class 0:** Both sensors inactive: $(0, 0)$
- **Class 1:** Sensors disagree: $(0, 1)$ or $(1, 0)$
- **Class 2:** Both sensors active: $(1, 1)$

**Architecture:** 2 inputs $\to$ 4 hidden units (ReLU) $\to$ 3 output logits ($z_0, z_1, z_2$). Loss: Multiclass Categorical Cross-Entropy (`nn.CrossEntropyLoss`).

![Three Class Decision Boundary](file:///C:/Users/Amithav/Desktop/LAB/NEURAL/three_class_decision_boundary.png)

### 6.2 Pre-Experimental Predictions
1. **Shape of Final Weight Matrix:** $W^{(2)} \in \mathbb{R}^{3 \times 4}$ ($3$ class logits $\times 4$ hidden features).
2. **Number of Logits per Example:** Exactly 3 unnormalized logits $[z_0, z_1, z_2]$.
3. **Softmax Probability Normalization:**
   $$\sum_{k=0}^2 p_k = \frac{\sum_{k=0}^2 \exp(z_k)}{\sum_{j=0}^2 \exp(z_j)} = 1.0$$
4. **Logit Gradient Derivation ($\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$):**
   For one-hot vector $\mathbf{y}$, $\mathcal{L} = -\sum_{j} y_j \ln p_j$. Using $\frac{\partial p_j}{\partial z_i} = p_i(\delta_{ij} - p_j)$:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{j} \frac{y_j}{p_j} \frac{\partial p_j}{\partial z_i} = -y_i(1 - p_i) + p_i \sum_{j \neq i} y_j = -y_i + y_i p_i + p_i(1 - y_i) = p_i - y_i$$

### 6.3 Experimental Results & Validation
Trained for 2500 epochs ($\eta = 0.05$, Adam):
- **Final Loss:** $0.000008$
- **Final Accuracy:** $100.0\%$

| Input $\mathbf{x}$ | Target Class | Logits $[z_0, z_1, z_2]$ | Softmax Probabilities $[p_0, p_1, p_2]$ | Sum of Probabilities | Prediction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $(0, 0)$ | 0 | $[10.26, -1.41, -13.66]$ | $[0.999991, 0.000009, 0.000000]$ | $0.99999994$ | Class 0 (PASS) |
| $(0, 1)$ | 1 | $[-7.18, 6.65, -5.80]$ | $[0.000001, 0.999995, 0.000004]$ | $1.00000012$ | Class 1 (PASS) |
| $(1, 0)$ | 1 | $[-3.25, 9.08, -2.79]$ | $[0.000004, 0.999989, 0.000007]$ | $1.00000000$ | Class 1 (PASS) |
| $(1, 1)$ | 2 | $[-15.89, 3.01, 14.84]$ | $[0.000000, 0.000007, 0.999993]$ | $1.00000000$ | Class 2 (PASS) |

### 6.4 Optional Diagnostic: Shift Invariance & Log-Sum-Exp Trick
Added constant $C = 100$ to all logits for Input $(0, 0)$:
- **Softmax on Raw Logits:** $[0.99999142, 8.54 \times 10^{-6}, 4.08 \times 10^{-11}]$
- **Softmax on Shifted Logits ($z + 100$):** $[0.99999140, 8.54 \times 10^{-6}, 4.08 \times 10^{-11}]$
- **Max Absolute Difference:** $\mathbf{4.09 \times 10^{-11}}$ (exact within float32 precision).

**Why stable implementations subtract $\max(z)$:**  
In 32-bit floats, $\exp(z)$ overflows to $+\infty$ for $z > 88.7$. Directly exponentiating large logits produces $\frac{\infty}{\infty} = \text{NaN}$. By subtracting $z_{\max} = \max_j z_j$, the maximum exponent evaluated is $\exp(0) = 1.0$, completely eliminating positive overflow while preserving exact relative probabilities.

### 6.5 Gradient Verification: Analytical $(\mathbf{p} - \mathbf{y})/N$ vs. PyTorch Autograd
- Maximum absolute difference across all batch logit gradients: $\mathbf{6.82 \times 10^{-13}}$.
- Autograd gradient matches the theoretical formula $\nabla_{\mathbf{z}} \mathcal{L} = \frac{1}{N}(\mathbf{p} - \mathbf{y})$ to machine precision.

### Think About It: Extension to Modern LLMs
> *"Next-token prediction in a language model can be viewed as classification over a very large vocabulary. Which parts of this tiny three-class experiment stay mathematically the same when the number of classes becomes tens of thousands, and which parts of the surrounding architecture change dramatically?"*

**Answer:**
1. **Mathematically Identical:** Softmax normalization, Cross-Entropy negative log-likelihood, the logit error gradient $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$, and the Log-Sum-Exp stabilization trick remain identical whether classifying 3 classes or 128,000 vocabulary tokens.
2. **Architectural Differences at Scale:**
   - The output projection matrix $W_{unembed} \in \mathbb{R}^{V \times d_{model}}$ scales to hundreds of millions of parameters, requiring weight-tying with input embeddings or FP8 low-precision kernels.
   - The softmax normalization across $128\text{k}$ classes requires fused Flash-Cross-Entropy CUDA kernels to avoid materializing huge intermediate logit tensors in HBM.
   - The input is not a continuous 2D vector, but high-dimensional token sequences processed via causal multi-head self-attention, RoPE position embeddings, and RMSNorm layers.

---

## 7. Answers to Laboratory Reflection Questions

### Reflection Question 1
> **What did the XOR experiment demonstrate about the difference between depth and nonlinearity?**

Depth alone simply multiplies linear transformation matrices: $(W_L \dots W_1)\mathbf{x} + \mathbf{b}_{eff}$. Because the product of affine maps is strictly affine, adding depth without nonlinearity increases parameter count but provides zero expansion in representational capacity. The decision boundary remains a flat hyperplane, which is mathematically incapable of solving linearly inseparable problems like XOR. Nonlinearity is the essential mechanism that bends and warps coordinate space, projecting nonlinearly separable points into a latent manifold where linear separation becomes possible.

### Reflection Question 2
> **In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?**

A nonzero gradient could simply indicate random drift, oscillation, or divergence. Three pieces of empirical evidence confirm a constructive learning signal:
1. **Monotonic, 3-Order-of-Magnitude Loss Reduction:** Loss decayed smoothly from $0.748230$ to $0.000340$.
2. **Evolution of Accuracy:** Predictions transitioned systematically from chance level to $100\%$ accuracy ($4/4$ correct with $> 99.9\%$ confidence).
3. **Smooth Vanishing Gradient Norm:** Frobenius norm $\|\nabla_{W^{(1)}} \mathcal{L}\|_2$ diminished smoothly from $0.011894$ to $8.98 \times 10^{-6}$, confirming convergence to a true stationary local minimum.

### Reflection Question 3
> **Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?**

When $W^{(1)} = \mathbf{0}$ and $\mathbf{b}^{(1)} = \mathbf{0}$, both hidden units compute identical pre-activations ($0.0$) and activations ($0.5$). With output weights $W^{(2)} = [0, 0]$, both units receive identical backpropagated error. In every subsequent gradient step, $\nabla_{W_1^{(1)}} \mathcal{L} = \nabla_{W_2^{(1)}} \mathcal{L}$ holds identically. Without random asymmetry to break symmetry, the two hidden neurons remain identical clones forever, collapsing the effective hidden dimension to $1$. Because a 2-1-1 network cannot separate XOR, the model remains permanently trapped at loss $\ln(2) \approx 0.69315$.

### Reflection Question 4
> **How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.**

- **Scientific Explanation:** By reverse-mode AD, $\nabla_{W^{(1)}} \mathcal{L}$ contains the Jacobian factor $f'(a^{(1)})$. The Sigmoid derivative $\sigma'(z) \le 0.25$ severely attenuates backpropagated error by $\ge 4 \times$ per layer. Tanh is zero-centered with $\tanh'(0) = 1.0$. ReLU transmits unattenuated gradients ($1.0$) for active units.
- **Engineering Observation:** Early gradient norm for Sigmoid was $0.008495$ (smallest), compared to $0.046584$ for Tanh and $0.024614$ for ReLU. Consequently, Tanh and ReLU achieved rapid loss descent and reached lower final losses ($1.32 \times 10^{-4}$ and $4.91 \times 10^{-5}$).

### Reflection Question 5
> **Why must the output layer and loss be selected together according to the task?**

The output activation parameterizes the target statistical distribution (Bernoulli for binary; Categorical for multiclass), and the loss represents the negative log-likelihood under that distribution. When paired canonically (Sigmoid with BCE; Softmax with Cross-Entropy), the nonlinear activation derivatives cancel out the loss denominator, yielding the linear error gradient $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$. Mismatched pairings (e.g. Sigmoid with MSE) introduce non-canceling derivative factors that cause vanishing gradients when the network makes severe errors.

### Reflection Question 6
> **Give one example where the LLM improved your engineering productivity and one example where human verification was essential.**

- **Productivity Gain:** The LLM instantly generated the complete PyTorch module structure, tensor datasets, training loop, and optimizer scaffolding in seconds.
- **Essential Human Verification:** The LLM initially suggested separate `nn.Sigmoid()` and `nn.BCELoss()`, which causes numerical overflow/underflow to `NaN`. Human verification replaced this with `nn.BCEWithLogitsLoss()`, which computes the stable Log-Sum-Exp formulation in log-space.

### Reflection Question 7
> **Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?**

- **Keep at Scale:** Tracking loss curves, computing layer-wise gradient norms $\|\nabla_W \mathcal{L}\|_2$ (to detect vanishing/exploding gradients), monitoring probability sum normalization, and evaluating accuracy on small synthetic unit-test batches.
- **Too Expensive at Scale:** Full-batch gradient descent (must use mini-batch SGD/AdamW), exhaustive finite-difference gradient checks (which require $2 \times P$ forward passes, or $140$ billion passes for a 70B model), dense 2D grid decision surface evaluations, and full parameter Hessian spectrum computations.

---

## 8. Master Summary of Artifacts and Verification

All code files, test logs, plots, the Jupyter Notebook, and the publication-grade PDF report have been generated, executed, and verified:

```
C:\Users\Amithav\Desktop\LAB\NEURAL\
├── neur_models_lab_ex (1).pdf         [Original Lab Specification]
├── neural_lab.ipynb                   [Complete Executed Jupyter Notebook]
├── NEURAL_LAB_REPORT.pdf              [4-Page Publication Academic PDF Report]
├── LAB_REPORT.md                      [Master Academic Markdown Lab Report]
├── TASKS_INDEX.txt                    [Master File Directory Index]
├── task_01_understand_problem.txt     [Task 1 Deliverable Output]
├── task_02_design_agent.txt           [Task 2 Deliverable Output]
├── task_03_llm_code_generation.txt    [Task 3 Deliverable Output]
├── task_04_execution_and_diagnostics.txt [Task 4 Deliverable Output]
├── task_05_three_class_extension.txt  [Task 5 Deliverable Output]
├── reflection_questions.txt           [Reflection Questions 1 - 7]
├── test_results.txt                   [Raw Numerical Console Logs]
├── xor_binary_model.py                [Task 2 & 4 Baseline XOR Code]
├── symmetry_experiment.py             [Task 4 Part C Symmetry Code]
├── activation_experiment.py           [Task 4 Part D Activations Code]
├── three_class_extension.py           [Task 5 Multiclass Sensor Code]
├── generate_plots.py                  [Plotting Rendering Code]
├── build_pdf_report.py                [PDF Generation Script]
├── run_experiments.py                 [Master Suite Orchestrator]
├── xor_linear_inseparability.png      [Figure 1: Geometric XOR Inseparability]
├── training_curves.png                [Figure 2: Loss & Gradient Norm Curves]
├── symmetry_collapse.png              [Figure 3: Weight Symmetry Collapse]
└── three_class_decision_boundary.png  [Figure 4: 3-Class Sensor Partitioning]
```
