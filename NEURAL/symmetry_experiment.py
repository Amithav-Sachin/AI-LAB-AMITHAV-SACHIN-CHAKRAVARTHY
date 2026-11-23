"""
symmetry_experiment.py
Laboratory - Neural Models: Learning, Depth, Activations, and Output Layers
Task 4, Part C: Symmetry Experiment (Zero Weight Initialization)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class XORNetZeroInit(nn.Module):
    def __init__(self, activation_fn=nn.Sigmoid):
        super(XORNetZeroInit, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.act = activation_fn()
        self.fc2 = nn.Linear(2, 1)

        # Zero-initialize all weights and biases explicitly
        nn.init.constant_(self.fc1.weight, 0.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.weight, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        h = self.act(self.fc1(x))
        logit = self.fc2(h)
        return logit, h


def run_symmetry_experiment(epochs=100, lr=0.1):
    print("=" * 70)
    print("TASK 4 PART C: SYMMETRY EXPERIMENT (ZERO INITIALIZATION)")
    print("=" * 70)

    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]], dtype=torch.float32)

    model = XORNetZeroInit(activation_fn=nn.Sigmoid)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    inspection_steps = [0, 1, 2, 5, 10, 20, 50, 100]
    records = []

    print(f"Tracking Row 1 and Row 2 of W^(1) over training steps:\n")
    print(f"{'Step':<6} | {'W^(1) Row 1 [w11, w12]':<25} | {'W^(1) Row 2 [w21, w22]':<25} | {'Identical?':<10} | {'Loss':<8}")
    print("-" * 80)

    for step in range(epochs + 1):
        # Inspect at specified steps
        if step in inspection_steps:
            w1 = model.fc1.weight.detach().numpy()
            row1 = w1[0, :].copy()
            row2 = w1[1, :].copy()
            is_identical = np.allclose(row1, row2, atol=1e-7)

            with torch.no_grad():
                logits, _ = model(X)
                cur_loss = criterion(logits, y).item()

            records.append({
                'step': step,
                'row1': row1,
                'row2': row2,
                'identical': is_identical,
                'loss': cur_loss
            })

            r1_str = f"[{row1[0]:.6f}, {row1[1]:.6f}]"
            r2_str = f"[{row2[0]:.6f}, {row2[1]:.6f}]"
            print(f"{step:<6} | {r1_str:<25} | {r2_str:<25} | {str(is_identical):<10} | {cur_loss:<.6f}")

        if step < epochs:
            optimizer.zero_grad()
            logits, h = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Final evaluation of predictions
    with torch.no_grad():
        final_logits, _ = model(X)
        final_probs = torch.sigmoid(final_logits).numpy().flatten()
        preds = (final_probs >= 0.5).astype(int)

    print("-" * 80)
    print("\nFinal Model Predictions under Zero Initialization:")
    for i in range(4):
        print(f"  Input: {tuple(X[i].numpy().astype(int))} -> Prob: {final_probs[i]:.4f}, Pred: {preds[i]}, Target: {int(y[i].item())}")

    print("\nSummary & Mathematical Diagnosis:")
    print("1. Are Row 1 and Row 2 of W^(1) identical at all steps? YES (strictly identical).")
    print("2. Why did this occur?")
    print("   Because W^(1) = 0 and b^(1) = 0, for any input x, both hidden neurons compute identical pre-activations:")
    print("     a_1 = W_1^(1) x + b_1 = 0,   a_2 = W_2^(1) x + b_2 = 0")
    print("   Hence, both neurons produce identical activations: h_1 = sigma(0) = 0.5, h_2 = sigma(0) = 0.5.")
    print("   Furthermore, because W^(2) = [0, 0], the backpropagated error to both hidden neurons is identical:")
    print("     dL/dh_1 = dL/dh_2 = (dL/dz) * W^(2) = 0")
    print("   Even when W^(2) updates, both components receive identical gradients, maintaining W_1^(2) = W_2^(2).")
    print("   Consequently, dL/dW_1^(1) == dL/dW_2^(1) throughout all gradient descent steps.")
    print("   The two hidden units remain clones of each other, collapsing the representational rank to 1.")
    print("   A 2-1-1 network cannot separate XOR; thus, the network is permanently trapped and cannot learn XOR.")
    print("=" * 70 + "\n")

    return records


if __name__ == '__main__':
    run_symmetry_experiment()
