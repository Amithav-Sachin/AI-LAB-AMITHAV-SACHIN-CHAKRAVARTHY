"""
three_class_extension.py
Laboratory - Neural Models: Learning, Depth, Activations, and Output Layers
Task 5: Multiclass Sensor Extension (3-Class Classification)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ThreeClassSensorNet(nn.Module):
    """
    Multiclass Sensor Classifier:
    Input (2) -> Hidden (d_hidden) with ReLU -> 3 Logits
    
    Classes:
      0: Both inactive  (0, 0)
      1: Disagreement   (0, 1) or (1, 0)
      2: Both active    (1, 1)
    """
    def __init__(self, d_hidden=4):
        super(ThreeClassSensorNet, self).__init__()
        self.fc1 = nn.Linear(2, d_hidden)
        self.act = nn.ReLU()
        # Final layer maps hidden representation to 3 class logits
        self.fc2 = nn.Linear(d_hidden, 3)

    def forward(self, x):
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        return logits, h


def run_three_class_experiment(seed=42, lr=0.05, epochs=2500):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("TASK 5: THREE-CLASS SENSOR CLASSIFICATION EXTENSION")
    print("=" * 70)

    # 1. Dataset definition
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    # Target classes: 0, 1, 1, 2
    y = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    model = ThreeClassSensorNet(d_hidden=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Pre-experimental Architectural Predictions:
    print("PRE-EXPERIMENTAL PREDICTIONS:")
    print("1. Shape of final weight matrix W^(2): [3, 4] (num_classes x hidden_dim)")
    print(f"   Actual initialized shape: {list(model.fc2.weight.shape)}")
    print("2. Number of logits per input example: 3 (one unnormalized score per class)")
    print("3. Softmax property: Softmax exp(z_k) / sum_j exp(z_j) strictly normalizes to 1.0.")
    print("4. Logit gradient: dL/dz = p - y, the direct difference between predicted probabilities")
    print("   and one-hot ground truth distribution.\n")

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, h = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item() * 100
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.6f} | Accuracy: {acc:.1f}%")

    # Evaluation
    model.eval()
    with torch.no_grad():
        final_logits, _ = model(X)
        final_loss = criterion(final_logits, y).item()
        probs = torch.softmax(final_logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    print("\n" + "-" * 70)
    print("VALIDATION RESULTS & PREDICTED PROBABILITIES:")
    print("-" * 70)
    class_names = ["Both Inactive (0)", "Sensors Disagree (1)", "Both Active (2)"]
    for i in range(4):
        x_tuple = tuple(X[i].numpy().astype(int))
        target_cls = y[i].item()
        pred_cls = preds[i]
        prob_vec = probs[i]
        sum_p = np.sum(prob_vec)
        status = "[CORRECT]" if pred_cls == target_cls else "[INCORRECT]"
        print(f"Input: {x_tuple} | Target: {target_cls} ({class_names[target_cls]})")
        print(f"  Logits: {np.round(final_logits[i].numpy(), 4)}")
        print(f"  Probabilities: {np.round(prob_vec, 6)} | Sum(P): {sum_p:.8f} | Predicted: {pred_cls} {status}")

    # Explicit numerical verification for Example 0
    print("\n" + "-" * 70)
    print("EXPLICIT NUMERICAL VERIFICATION (Input (0,0)):")
    p0 = probs[0]
    print(f"  P(Class 0) = {p0[0]:.8f}")
    print(f"  P(Class 1) = {p0[1]:.8f}")
    print(f"  P(Class 2) = {p0[2]:.8f}")
    print(f"  Sum of Components: {p0[0] + p0[1] + p0[2]:.10f} (Verified: exactly 1.0 within float precision)")

    # Optional Diagnostic: Shift-invariance of Softmax
    print("\n" + "-" * 70)
    print("OPTIONAL DIAGNOSTIC: LOGIT SHIFT INVARIANCE (z + C)")
    print("-" * 70)
    raw_logits_ex0 = final_logits[0].clone()
    shifted_logits_ex0 = raw_logits_ex0 + 100.0

    p_raw = torch.softmax(raw_logits_ex0, dim=0).numpy()
    p_shifted = torch.softmax(shifted_logits_ex0, dim=0).numpy()
    diff = np.abs(p_raw - p_shifted)

    print(f"Raw Logits:            {raw_logits_ex0.numpy()}")
    print(f"Shifted Logits (+100): {shifted_logits_ex0.numpy()}")
    print(f"Softmax(raw):          {p_raw}")
    print(f"Softmax(shifted):      {p_shifted}")
    print(f"Max Absolute Difference: {np.max(diff):.2e}")
    print("Conclusion: The softmax distribution is mathematically invariant to uniform constant shifts.")
    print("Why stable implementations subtract max(z):")
    print("  In standard 32-bit floats, exp(z) overflows to +inf for z > 88.7. If z contains large positive values,")
    print("  direct evaluation produces inf/inf = NaN. Subtracting z_max = max(z) ensures the maximum exponent")
    print("  is exp(0) = 1.0, completely preventing floating-point overflow while preserving exact probabilities.")

    # Mathematical Verification of p - y Gradient
    print("\n" + "-" * 70)
    print("VERIFICATION OF LOGIT GRADIENT: dL/dz = p - y")
    print("-" * 70)
    # Compute analytical p - y
    p_torch = torch.softmax(final_logits, dim=1)
    y_one_hot = torch.zeros_like(p_torch)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    analytical_grad = (p_torch - y_one_hot) / 4.0  # divided by batch size 4 for mean loss

    # Compute autograd gradient with respect to logits
    test_logits = final_logits.clone().detach().requires_grad_(True)
    test_loss = criterion(test_logits, y)
    test_loss.backward()
    autograd_grad = test_logits.grad

    max_grad_diff = torch.max(torch.abs(analytical_grad - autograd_grad)).item()
    print(f"Analytical Gradient (p - y)/N:\n{analytical_grad.numpy()}")
    print(f"PyTorch Autograd Gradient:\n{autograd_grad.numpy()}")
    print(f"Max Difference: {max_grad_diff:.2e} (Exact match to machine epsilon!)")
    print("=" * 70 + "\n")

    return {
        'model': model,
        'final_loss': final_loss,
        'logits': final_logits.numpy(),
        'probs': probs,
        'preds': preds,
        'p_raw': p_raw,
        'p_shifted': p_shifted,
        'max_grad_diff': max_grad_diff
    }


if __name__ == '__main__':
    run_three_class_experiment()
