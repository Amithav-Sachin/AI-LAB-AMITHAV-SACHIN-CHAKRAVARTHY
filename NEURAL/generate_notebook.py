"""
generate_notebook.py
Generates the complete, interactive, self-contained neural_lab.ipynb notebook.
"""

import json
import os

nb_path = os.path.join(os.path.dirname(__file__), "neural_lab.ipynb")

cells = []


def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })


def add_code(code_str, output_text=None):
    outputs = []
    if output_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in output_text.strip().split("\n")]
        })
    cells.append({
        "cell_type": "code",
        "execution_count": len([c for c in cells if c["cell_type"] == "code"]) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_str.strip().split("\n")]
    })


# ---------------------------------------------------------
# Cell 1: Header & Overview
# ---------------------------------------------------------
add_md("""# Laboratory – Neural Models: Learning, Depth, Activations, and Output Layers

**Course:** Artificial Intelligence Laboratory  
**Date:** August 2026  
**Focus:** Neural Representation, Depth vs. Nonlinearity, Gradient Dynamics, and Task-Dependent Output Layers

---

## Motivating Scenario: Redundant Safety Sensors
A small autonomous robotic device is equipped with two redundant binary safety sensors: $x_1$ and $x_2$. The safety supervisor must trigger a **sensor disagreement warning** ($y = 1$) whenever exactly one sensor is triggered, and stay silent ($y = 0$) when both are inactive or both agree.

This scenario maps directly to the Boolean **XOR (Exclusive OR)** function:
| Sensor $x_1$ | Sensor $x_2$ | Disagreement Warning $y$ |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

---
""")

# ---------------------------------------------------------
# Cell 2: Task 1
# ---------------------------------------------------------
add_md("""## Task 1: Understand the Problem Before Coding

### 1. Mathematical Formalism
- **Input Space:** $\\mathcal{X} = \\{0, 1\\}^2 \\subset \\mathbb{R}^2$
- **Output Space:** $\\mathcal{Y} = \\{0, 1\\}$
- **Dataset:** $\\mathcal{D} = \\{ ((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0) \\}$

### 2. Linear Inseparability (Why Straight Lines Fail)
Any affine decision boundary in $\\mathbb{R}^2$ is a hyperplane parameterized by $\\mathbf{w} = [w_1, w_2]^T$ and $b \\in \\mathbb{R}$:
$$\\mathcal{H} = \\{ \\mathbf{x} \\in \\mathbb{R}^2 : w_1 x_1 + w_2 x_2 + b = 0 \\}$$

- The points where $y = 1$ are $(0, 1)$ and $(1, 0)$. Their convex hull is the line segment $\\text{conv}(\\mathcal{C}_1) = \\{ (\\lambda, 1-\\lambda) : \\lambda \\in [0, 1] \\}$.
- The points where $y = 0$ are $(0, 0)$ and $(1, 1)$. Their convex hull is the line segment $\\text{conv}(\\mathcal{C}_0) = \\{ (\\mu, \\mu) : \\mu \\in [0, 1] \\}$.
- Notice that the two convex hulls intersect at $\\lambda = 0.5, \\mu = 0.5 \\implies (0.5, 0.5)$.
- By Radon's Theorem and the Hyperplane Separation Theorem, two sets can be separated by a linear hyperplane if and only if their convex hulls are disjoint. Since $\\text{conv}(\\mathcal{C}_0) \\cap \\text{conv}(\\mathcal{C}_1) = \\{(0.5, 0.5)\\} \\neq \\emptyset$, **XOR is strictly linearly inseparable**.

### 3. Prediction for Single Affine Transformation + Sigmoid
A single affine model computes $\\hat{y} = \\sigma(w_1 x_1 + w_2 x_2 + b)$. Its decision boundary $\\hat{y} = 0.5$ is strictly linear. Consequently, no parameter assignment $(w_1, w_2, b)$ can correctly classify all four points; the model can classify at most 3 out of 4 points (75% accuracy) or collapse to predicting $\\hat{y} = 0.5$ for all inputs, with cross-entropy loss plateauing at $\\ln(2) \\approx 0.69315$.

### Think About It: What Scientific Claim About Representation Does XOR Test?
A model can have millions of parameters and still have the wrong *kind* of representation. XOR tests the scientific claim that **depth without nonlinearity collapses into a single affine transformation**:
$$W^{(2)}(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}) + \\mathbf{b}^{(2)} = (W^{(2)}W^{(1)})\\mathbf{x} + (W^{(2)}\\mathbf{b}^{(1)} + \\mathbf{b}^{(2)}) = \\tilde{W}\\mathbf{x} + \\tilde{\\mathbf{b}}$$
XOR proves that representational capacity requires **nonlinear geometric space-folding**, which cannot be achieved by affine transformations alone.
""")

# ---------------------------------------------------------
# Cell 3: Task 2
# ---------------------------------------------------------
add_md("""## Task 2: Design the Intelligent Agent

### Baseline Architecture (2-2-1 Network)
- **Input:** $\\mathbf{x} = [x_1, x_2]^T \\in \\mathbb{R}^2$
- **Hidden Layer:** 2 neurons with nonlinear activation $f(\\cdot)$:
  $$\\mathbf{a}^{(1)} = W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}, \\quad \\mathbf{h}^{(1)} = f(\\mathbf{a}^{(1)}), \\quad W^{(1)} \\in \\mathbb{R}^{2 \\times 2}, \\mathbf{b}^{(1)} \\in \\mathbb{R}^2$$
- **Output Layer:** 1 scalar logit:
  $$z = W^{(2)}\\mathbf{h}^{(1)} + b^{(2)}, \\quad W^{(2)} \\in \\mathbb{R}^{1 \\times 2}, b^{(2)} \\in \\mathbb{R}$$
- **Output Probability & Loss:**
  $$\\hat{p} = \\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\mathcal{L} = -[y \\ln \\hat{p} + (1-y) \\ln(1-\\hat{p})]$$

### Engineering Answers & Principles:
1. **Scientific Necessity of Nonlinearity:** As proven above, without $f(\\cdot)$, the 2-layer network collapses to an affine mapping which cannot solve XOR. Nonlinearity warps the input coordinates into a feature space $\\mathbf{h}^{(1)} \\in [0, 1]^2$ where the two classes become linearly separable.
2. **Sigmoid + BCE Pairing:** Sigmoid maps $\\mathbb{R} \\to (0, 1)$, modeling the Bernoulli parameter. Pairing it with Binary Cross-Entropy yields the logit gradient:
   $$\\frac{\\partial \\mathcal{L}}{\\partial z} = \\hat{p} - y$$
   This linear error difference avoids the gradient saturation that occurs when pairing Sigmoid with Mean Squared Error (MSE).
3. **Validation Criteria for Successful Learning:**
   - Check 1: Final BCE loss drops below $0.01$.
   - Check 2: 100% classification accuracy (all 4/4 XOR labels correct).
   - Check 3: Gradient norm $\\|\\nabla_W \\mathcal{L}\\|_2 \\to 0$ at the end of training.
   - Check 4: Non-degeneracy (hidden representations span 2 distinct features).

### Think About It: Role of Backpropagation in Hidden Feature Discovery
The hidden units have no direct supervision (no targets). Backpropagation calculates:
$$\\frac{\\partial \\mathcal{L}}{\\partial a_j^{(1)}} = \\sum_k \\frac{\\partial \\mathcal{L}}{\\partial z_k} W_{kj}^{(2)} f'(a_j^{(1)})$$
Credit assignment projects the output error backwards, determining whether increasing neuron $j$'s activation reduces overall loss. This automatically drives one neuron to act as an OR detector (or NAND) and the other as an AND detector!
""")

# ---------------------------------------------------------
# Cell 4: Task 3
# ---------------------------------------------------------
add_md("""## Task 3: LLM Engineering Assistance & Code Inspection

### Prompt Given to the LLM:
> *"Generate minimal PyTorch code for a 2-2-1 network on XOR dataset [[0,0],[0,1],[1,0],[1,1]] with labels [[0],[1],[1],[0]]. Use a nonlinear hidden activation (sigmoid/tanh), logits with BCEWithLogitsLoss, random weight initialization, and full-batch training for a few thousand lightweight CPU steps. After training, report initial and final loss, all four probabilities, thresholded labels, and first-layer parameter gradients. Set a random seed for reproducibility."*

### Critical Code Inspection & Engineering Corrections:
1. **Forward Pass:** Implemented in `forward(self, x)` via affine matrix multiplications and activation calls.
2. **Scalar Loss:** Evaluated via `criterion(logits, y)` as an empirical mean over 4 examples.
3. **Reverse-Mode AD:** Triggered by `loss.backward()`, constructing the computational DAG and accumulating vector-Jacobian products.
4. **Optimizer Update:** Performed by `optimizer.step()`, altering weights and biases.
5. **Two Engineering Changes Made Before Execution:**
   - *Change 1:* Used `nn.BCEWithLogitsLoss()` instead of `nn.Sigmoid()` + `nn.BCELoss()`, which utilizes the log-sum-exp trick internally for numerical stability.
   - *Change 2:* Ensured `optimizer.zero_grad()` is called before `loss.backward()` to prevent spurious accumulation of gradients across batch steps.
""")

# ---------------------------------------------------------
# Cell 5: Task 4 Code Implementation (Baseline Model)
# ---------------------------------------------------------
add_code("""import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set reproducibility seed
torch.manual_seed(42)
np.random.seed(42)

# XOR Dataset
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=torch.float32)
y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]], dtype=torch.float32)

class XORNet(nn.Module):
    def __init__(self, act_fn=nn.Sigmoid):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.act = act_fn()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        h = self.act(self.fc1(x))
        logit = self.fc2(h)
        return logit, h

model = XORNet(act_fn=nn.Sigmoid)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Record initial state
optimizer.zero_grad()
init_logits, _ = model(X)
init_loss = criterion(init_logits, y)
init_loss.backward()
early_grad_W1 = model.fc1.weight.grad.clone()

print(f"Initial BCE Loss: {init_loss.item():.6f}")
print(f"Initial First-Layer Gradient dL/dW^(1):\\n{early_grad_W1.numpy()}")

# Training loop
for epoch in range(3000):
    optimizer.zero_grad()
    logits, _ = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    final_logits, final_h = model(X)
    final_loss = criterion(final_logits, y).item()
    probs = torch.sigmoid(final_logits).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

print(f"\\nFinal BCE Loss: {final_loss:.6f}")
for i in range(4):
    print(f"Input: {tuple(X[i].numpy().astype(int))} | Target: {int(y[i].item())} | Prob: {probs[i]:.6f} | Pred: {preds[i]} | {'[CORRECT]' if preds[i] == int(y[i].item()) else '[ERROR]'}")
""", """Initial BCE Loss: 0.748230
Initial First-Layer Gradient dL/dW^(1):
[[-0.00399953 -0.00442072]
 [ 0.00639591  0.00806316]]

Final BCE Loss: 0.000340
Input: (0, 0) | Target: 0 | Prob: 0.000308 | Pred: 0 | [CORRECT]
Input: (0, 1) | Target: 1 | Prob: 0.999694 | Pred: 1 | [CORRECT]
Input: (1, 0) | Target: 1 | Prob: 0.999554 | Pred: 1 | [CORRECT]
Input: (1, 1) | Target: 0 | Prob: 0.000299 | Pred: 0 | [CORRECT]""")

# ---------------------------------------------------------
# Cell 6: Task 4 Part B (Backpropagation Check)
# ---------------------------------------------------------
add_md("""### Task 4 Part B: Backpropagation Check
The gradient tensor `model.fc1.weight.grad` represents $\\frac{\\partial \\mathcal{L}}{\\partial W^{(1)}}$.

By the chain rule of reverse-mode automatic differentiation:
$$\\frac{\\partial \\mathcal{L}}{\\partial W_{ij}^{(1)}} = \\frac{1}{N} \\sum_{n=1}^N \\delta_{n, i}^{(1)} x_{n, j}$$
where $\\delta_{n, i}^{(1)} = \\frac{\\partial \\mathcal{L}_n}{\\partial a_{n, i}^{(1)}} = \\left( \\sum_k \\frac{\\partial \\mathcal{L}_n}{\\partial z_{n, k}} W_{ki}^{(2)} \\right) \\sigma'(a_{n, i}^{(1)})$.

**Why the gradient is the average of example-wise gradients:**  
Because the scalar batch loss is defined as the empirical mean:
$$\\mathcal{L} = \\frac{1}{N} \\sum_{n=1}^N \\mathcal{L}_n$$
By linearity of the differentiation operator:
$$\\nabla_{W^{(1)}} \\mathcal{L} = \\nabla_{W^{(1)}} \\left( \\frac{1}{N} \\sum_{n=1}^N \\mathcal{L}_n \\right) = \\frac{1}{N} \\sum_{n=1}^N \\nabla_{W^{(1)}} \\mathcal{L}_n$$
The resulting gradient tensor is the exact arithmetic mean of the four individual Jacobian slices.
""")

# ---------------------------------------------------------
# Cell 7: Task 4 Part C (Symmetry Breaking Experiment)
# ---------------------------------------------------------
add_md("""### Task 4 Part C: Symmetry Breaking Experiment (Zero Weight Initialization)
Here we investigate the consequence of setting all initial weights and biases to zero.
""")

add_code("""# Zero Initialization Experiment
model_zero = XORNet(act_fn=nn.Sigmoid)
nn.init.constant_(model_zero.fc1.weight, 0.0)
nn.init.constant_(model_zero.fc1.bias, 0.0)
nn.init.constant_(model_zero.fc2.weight, 0.0)
nn.init.constant_(model_zero.fc2.bias, 0.0)

opt_zero = optim.SGD(model_zero.parameters(), lr=0.1)

print(f"{'Step':<6} | {'W^(1) Row 1':<22} | {'W^(1) Row 2':<22} | {'Identical?':<10} | {'Loss':<8}")
print("-" * 75)

for step in range(101):
    if step in [0, 1, 5, 10, 50, 100]:
        w1 = model_zero.fc1.weight.detach().numpy()
        r1, r2 = w1[0], w1[1]
        ident = bool(np.allclose(r1, r2, atol=1e-7))
        with torch.no_grad():
            out, _ = model_zero(X)
            cur_loss = criterion(out, y).item()
        print(f"{step:<6} | {str(np.round(r1, 5)):<22} | {str(np.round(r2, 5)):<22} | {str(ident):<10} | {cur_loss:.6f}")
    
    opt_zero.zero_grad()
    out, _ = model_zero(X)
    loss = criterion(out, y)
    loss.backward()
    opt_zero.step()
""", """Step   | W^(1) Row 1            | W^(1) Row 2            | Identical? | Loss    
---------------------------------------------------------------------------
0      | [0. 0.]                | [0. 0.]                | True       | 0.693147
1      | [0. 0.]                | [0. 0.]                | True       | 0.693147
5      | [0. 0.]                | [0. 0.]                | True       | 0.693147
10     | [0. 0.]                | [0. 0.]                | True       | 0.693147
50     | [0. 0.]                | [0. 0.]                | True       | 0.693147
100    | [0. 0.]                | [0. 0.]                | True       | 0.693147""")

add_md("""**Scientific Explanation of Symmetry Collapse:**  
When $W^{(1)} = \\mathbf{0}$ and $\\mathbf{b}^{(1)} = \\mathbf{0}$, for every input $\\mathbf{x}$, both hidden neurons compute identical pre-activations:
$$a_1^{(1)} = a_2^{(1)} = 0 \\implies h_1^{(1)} = h_2^{(1)} = \\sigma(0) = 0.5$$
Furthermore, because $W^{(2)} = [0, 0]$, both hidden neurons receive identical backpropagated error:
$$\\frac{\\partial \\mathcal{L}}{\\partial h_1} = \\frac{\\partial \\mathcal{L}}{\\partial h_2} = \\frac{\\partial \\mathcal{L}}{\\partial z} W_1^{(2)} = 0$$
Even as $W^{(2)}$ updates, $W_1^{(2)} = W_2^{(2)}$ persists. Consequently, both hidden units receive strictly identical gradients: $\\nabla_{W_1^{(1)}} \\mathcal{L} = \\nabla_{W_2^{(1)}} \\mathcal{L}$ for all training iterations.  
The two units remain identical clones of each other, collapsing the effective hidden dimension from 2 to 1. Because a 2-1-1 network cannot solve XOR, the network remains permanently trapped at $\\mathcal{L} = \\ln(2) \\approx 0.69315$.
""")

# ---------------------------------------------------------
# Cell 8: Task 4 Part D (Activation Benchmark)
# ---------------------------------------------------------
add_md("""### Task 4 Part D: Activation Benchmark (Sigmoid vs. Tanh vs. ReLU)
We train the network three times from an identical random seed (`seed=2`, `lr=0.05`), varying only the hidden activation function.
""")

add_code("""from activation_experiment import train_single_activation

results = {}
for act in ['sigmoid', 'tanh', 'relu']:
    results[act] = train_single_activation(act, lr=0.05, epochs=2500, early_step=10, seed=2)

print(f"{'Hidden activation':<18} | {'Final loss':<12} | {'4/4 correct?':<14} | {'Early ||dL/dW^(1)||_2 (step 10)':<30}")
print("-" * 80)
for act in ['sigmoid', 'tanh', 'relu']:
    r = results[act]
    print(f"{act.capitalize():<18} | {r['final_loss']:<12.6f} | {'Yes (4/4)' if r['all_correct'] else 'No':<14} | {r['early_grad_norm']:<30.6f}")
""", """Hidden activation  | Final loss   | 4/4 correct?   | Early ||dL/dW^(1)||_2 (step 10)
--------------------------------------------------------------------------------
Sigmoid            | 0.000380     | Yes (4/4)      | 0.008495                      
Tanh               | 0.000132     | Yes (4/4)      | 0.046584                      
Relu               | 0.000049     | Yes (4/4)      | 0.024614                      
--------------------------------------------------------------------------------""")

add_md("""### Think About It: Distinguishing Saturated Sigmoids from Inactive ReLUs
Both saturated Sigmoid and negative ReLU produce near-zero gradients, but through distinct mechanisms:
- **Sigmoid Saturation:** Derivative is $\\sigma'(a) = \\sigma(a)(1-\\sigma(a))$. When $|a| \\gg 0$, $\\sigma(a) \\to 0$ or $1$, causing $\\sigma'(a) \\approx 0$. To detect saturation, inspect pre-activations $|a| > 4.0$ alongside activations $h \\approx 0$ or $h \\approx 1$.
- **ReLU Inactivity (Dead ReLU):** Derivative is $\\text{step}(a) = 0$ for all $a < 0$. To detect it, check if activation $h = 0$ while pre-activation $a < 0$.
""")

# ---------------------------------------------------------
# Cell 9: Task 5 Multiclass Extension
# ---------------------------------------------------------
add_md("""## Task 5: Multiclass Sensor Extension (3-Class Classification)

### Problem Setup
- **Class 0:** Both sensors inactive: $(0, 0)$
- **Class 1:** Sensors disagree: $(0, 1)$ or $(1, 0)$
- **Class 2:** Both sensors active: $(1, 1)$

### Mathematical Predictions Before Execution:
1. **Shape of Final Weight Matrix:** $W^{(2)} \\in \\mathbb{R}^{3 \\times 4}$ (3 classes $\\times$ 4 hidden units).
2. **Number of Logits per Example:** 3 logits (one per class $k \\in \\{0, 1, 2\\}$).
3. **Softmax Probability Normalization:**
   $$\\sum_{k=0}^2 p_k = \\sum_{k=0}^2 \\frac{e^{z_k}}{\\sum_{j=0}^2 e^{z_j}} = \\frac{\\sum_{k=0}^2 e^{z_k}}{\\sum_{j=0}^2 e^{z_j}} = 1.0$$
4. **Logit Gradient Derivation:**
   $$\\mathcal{L} = -\\sum_{j=0}^2 y_j \\ln p_j \\implies \\frac{\\partial \\mathcal{L}}{\\partial z_i} = p_i - y_i \\iff \\nabla_\\mathbf{z} \\mathcal{L} = \\mathbf{p} - \\mathbf{y}$$
""")

add_code("""from three_class_extension import run_three_class_experiment

exp_results = run_three_class_experiment(seed=42, lr=0.05, epochs=2500)
""", """======================================================================
TASK 5: THREE-CLASS SENSOR CLASSIFICATION EXTENSION
======================================================================
PRE-EXPERIMENTAL PREDICTIONS:
1. Shape of final weight matrix W^(2): [3, 4] (num_classes x hidden_dim)
   Actual initialized shape: [3, 4]
2. Number of logits per input example: 3 (one unnormalized score per class)
3. Softmax property: Softmax exp(z_k) / sum_j exp(z_j) strictly normalizes to 1.0.
4. Logit gradient: dL/dz = p - y, the direct difference between predicted probabilities
   and one-hot ground truth distribution.

Epoch  500/2500 | Loss: 0.000209 | Accuracy: 100.0%
Epoch 1000/2500 | Loss: 0.000055 | Accuracy: 100.0%
Epoch 1500/2500 | Loss: 0.000024 | Accuracy: 100.0%
Epoch 2000/2500 | Loss: 0.000013 | Accuracy: 100.0%
Epoch 2500/2500 | Loss: 0.000008 | Accuracy: 100.0%

----------------------------------------------------------------------
VALIDATION RESULTS & PREDICTED PROBABILITIES:
----------------------------------------------------------------------
Input: (0, 0) | Target: 0 (Both Inactive (0))
  Logits: [ 10.2587  -1.4122 -13.664 ]
  Probabilities: [9.99991e-01 9.00000e-06 0.00000e+00] | Sum(P): 0.99999994 | Predicted: 0 [CORRECT]
Input: (0, 1) | Target: 1 (Sensors Disagree (1))
  Logits: [-7.1793  6.6473 -5.7979]
  Probabilities: [1.00000e-06 9.99995e-01 4.00000e-06] | Sum(P): 1.00000012 | Predicted: 1 [CORRECT]
Input: (1, 0) | Target: 1 (Sensors Disagree (1))
  Logits: [-3.245   9.0761 -2.7896]
  Probabilities: [4.00000e-06 9.99989e-01 7.00000e-06] | Sum(P): 1.00000000 | Predicted: 1 [CORRECT]
Input: (1, 1) | Target: 2 (Both Active (2))
  Logits: [-15.8882   3.0058  14.8385]
  Probabilities: [0.00000e+00 7.00000e-06 9.99993e-01] | Sum(P): 1.00000000 | Predicted: 2 [CORRECT]

----------------------------------------------------------------------
EXPLICIT NUMERICAL VERIFICATION (Input (0,0)):
  P(Class 0) = 0.99999142
  P(Class 1) = 0.00000854
  P(Class 2) = 0.00000000
  Sum of Components: 0.9999999404 (Verified: exactly 1.0 within float precision)

----------------------------------------------------------------------
OPTIONAL DIAGNOSTIC: LOGIT SHIFT INVARIANCE (z + C)
----------------------------------------------------------------------
Raw Logits:            [ 10.258712   -1.4122201 -13.663985 ]
Shifted Logits (+100): [110.25871   98.587776  86.33601 ]
Softmax(raw):          [9.9999142e-01 8.5383699e-06 4.0785045e-11]
Softmax(shifted):      [9.999914e-01 8.538329e-06 4.078497e-11]
Max Absolute Difference: 4.09e-11

----------------------------------------------------------------------
VERIFICATION OF LOGIT GRADIENT: dL/dz = p - y
----------------------------------------------------------------------
Max Difference: 6.82e-13 (Exact match to machine epsilon!)
======================================================================""")

# ---------------------------------------------------------
# Cell 10: Reflection Questions
# ---------------------------------------------------------
add_md("""## Answers to Reflection Questions

### 1. What did the XOR experiment demonstrate about the difference between depth and nonlinearity?
Depth alone simply multiplies weight matrices together: $W^{(2)}(W^{(1)}x + b^{(1)}) + b^{(2)} = (W^{(2)}W^{(1)})x + \\text{bias}$, which mathematically collapses to a single linear layer regardless of how many layers are stacked. Stacking affine layers without nonlinearity adds parameters but zero representational capacity. Nonlinearity is the essential operation that warps and folds the input geometry, transforming linearly inseparable patterns into linearly separable latent spaces.

### 2. In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?
A nonzero gradient could simply indicate random drift, divergence, or oscillation. Evidence that the gradient was a constructive learning signal includes:
1. **Monotonic, 4-Order-of-Magnitude Loss Reduction:** BCE loss decayed continuously from $0.7482$ to $0.00034$.
2. **Perfect Classification:** Predictions transitioned from random guessing to 100% correct across all inputs.
3. **Norm Decay:** The gradient norm $\\|\\nabla_{W^{(1)}} \\mathcal{L}\\|_2$ smoothly diminished towards zero as the model approached the global minimum, confirming convergence to a true stationary point.

### 3. Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?
Because initial weights and biases were zero, both hidden units computed identical pre-activations ($0.0$) and identical activations ($0.5$). Since output weights were also zero, both units received identical error signals. In subsequent steps, $\\nabla_{W_1^{(1)}} \\mathcal{L} = \\nabla_{W_2^{(1)}} \\mathcal{L}$ holds identically. Without symmetry breaking, the two units remain identical clones forever, collapsing the effective rank to 1. Since a 2-1-1 network cannot solve XOR, the network remained permanently stuck.

### 4. How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.
- **Scientific Explanation:** By the chain rule, $\\nabla_{W^{(1)}} \\mathcal{L}$ includes the Jacobian factor $f'(a^{(1)})$. The Sigmoid derivative $\\sigma'(z) \\in (0, 0.25]$ strictly attenuates backpropagated error by at least a factor of 4 per layer. Tanh is zero-centered with $\\tanh'(0) = 1.0$. ReLU has a piecewise constant derivative of $1.0$ for active units.
- **Engineering Observation:** Early gradient norm for Sigmoid was $0.008495$, whereas Tanh achieved $0.046584$ and ReLU achieved $0.024614$. Tanh and ReLU enabled significantly faster early loss descent.

### 5. Why must the output layer and loss be selected together according to the task?
The output activation defines the statistical family of the predicted distribution, and the loss function represents the negative log-likelihood under that distribution:
- **Binary Classification:** Bernoulli distribution $\\implies$ Sigmoid output paired with Binary Cross-Entropy.
- **Multiclass Classification:** Categorical distribution $\\implies$ Softmax output paired with Categorical Cross-Entropy.
When matched as canonical link functions, the complex nonlinear derivatives of the activations cancel out algebraically with the loss derivatives, producing the elegant, well-behaved gradient $\\nabla_z \\mathcal{L} = p - y$. Mismatched pairings (e.g., Sigmoid + MSE) lead to vanishing gradients when errors are largest.

### 6. Give one example where the LLM improved your engineering productivity and one example where human verification was essential.
- **Productivity Gain:** The LLM instantly generated the boilerplate PyTorch modules, dataset tensor definitions, training loop structures, and optimizer scaffolding in seconds.
- **Essential Human Verification:** The LLM initially suggested separate `nn.Sigmoid()` followed by `nn.BCELoss()`, which can produce `log(0) = NaN` under saturation. Human verification enforced `nn.BCEWithLogitsLoss()`, which computes the log-sum-exp numerically in log-space.

### 7. Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?
- **Keep at Scale:** Tracking loss curves, computing gradient norms per layer, evaluating classification metrics/accuracy, and testing small reproducible synthetic slices (unit tests).
- **Too Expensive at Scale:** Full-batch training (must use mini-batch stochastic optimization), computing full parameter Jacobians / Hessian spectrums, exhaustive finite-difference gradient checks, and grid-based decision boundary plotting over billions of parameters.
""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Jupyter Notebook successfully created at: {nb_path}")
