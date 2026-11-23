"""
xor_binary_model.py
Laboratory - Neural Models: Learning, Depth, Activations, and Output Layers
Tasks 2, 3, 4A & 4B: Baseline 2-2-1 Neural Model for XOR Sensor Disagreement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class XORNet(nn.Module):
    """
    Minimal 2-2-1 Neural Network for XOR Classification.
    Architecture:
      Input (2) -> Hidden (2) with nonlinear activation -> Output (1) logit
    """
    def __init__(self, activation_fn=nn.Sigmoid):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.act = activation_fn()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        # Forward pass: affine -> nonlinear activation -> affine (logit)
        h = self.act(self.fc1(x))
        logit = self.fc2(h)
        return logit, h


def train_xor(activation_name='sigmoid', lr=0.1, epochs=3000, seed=42, verbose=True):
    """
    Trains the 2-2-1 XOR network using BCEWithLogitsLoss.
    Reports initial & final loss, predictions, and gradient tensors.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    act_dict = {
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'relu': nn.ReLU
    }
    act_cls = act_dict.get(activation_name.lower(), nn.Sigmoid)

    # XOR Dataset: 4 explicit labelled examples
    # x1: sensor 1, x2: sensor 2 -> y: disagreement warning (XOR)
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]], dtype=torch.float32)

    model = XORNet(activation_fn=act_cls)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initial forward pass to record initial loss and initial gradient
    optimizer.zero_grad()
    initial_logits, _ = model(X)
    initial_loss = criterion(initial_logits, y)
    initial_loss.backward()
    
    # Capture early gradient norm of W^(1)
    early_grad_W1 = model.fc1.weight.grad.clone()
    early_grad_norm_W1 = torch.norm(early_grad_W1).item()

    if verbose:
        print("=" * 60)
        print(f"TRAINING 2-2-1 XOR MODEL [Activation: {activation_name.upper()}]")
        print("=" * 60)
        print(f"Initial BCE Loss: {initial_loss.item():.6f}")
        print(f"Initial First-Layer Weight Gradient (dL/dW^(1)):\n{early_grad_W1.numpy()}")
        print(f"Initial dL/dW^(1) Frobenius Norm: {early_grad_norm_W1:.6f}\n")

    history = {'loss': [], 'grad_norm_w1': []}

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, h = model(X)
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = torch.norm(model.fc1.weight.grad).item()
        history['loss'].append(loss.item())
        history['grad_norm_w1'].append(grad_norm)

        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            probs = torch.sigmoid(logits).detach()
            preds = (probs >= 0.5).float()
            correct = (preds == y).sum().item()
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.6f} | Correct: {int(correct)}/4 | Grad Norm W1: {grad_norm:.6f}")

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        final_logits, final_hidden = model(X)
        final_loss = criterion(final_logits, y).item()
        final_probs = torch.sigmoid(final_logits).numpy().flatten()
        thresholded_preds = (final_probs >= 0.5).astype(int)
        true_labels = y.numpy().flatten().astype(int)
        all_correct = bool(np.array_equal(thresholded_preds, true_labels))

    # Also capture gradient at final step
    model.train()
    optimizer.zero_grad()
    final_logits_grad, _ = model(X)
    criterion(final_logits_grad, y).backward()
    final_grad_W1 = model.fc1.weight.grad.clone()

    if verbose:
        print("\n" + "-" * 60)
        print("FINAL RESULTS AND VERIFICATION:")
        print("-" * 60)
        print(f"Final BCE Loss: {final_loss:.6f}")
        print("Predictions on 4 XOR Inputs:")
        for i in range(4):
            x_in = tuple(X[i].numpy().astype(int))
            print(f"  Input: {x_in} | True y: {true_labels[i]} | Prob: {final_probs[i]:.6f} | Pred: {thresholded_preds[i]} | {'[CORRECT]' if thresholded_preds[i] == true_labels[i] else '[INCORRECT]'}")
        print(f"All 4/4 Correct: {all_correct}")
        print(f"\nFinal Layer 1 Weights (W^(1)):\n{model.fc1.weight.data.numpy()}")
        print(f"Final Layer 1 Biases (b^(1)):\n{model.fc1.bias.data.numpy()}")
        print(f"Final Layer 2 Weights (W^(2)):\n{model.fc2.weight.data.numpy()}")
        print(f"Final Layer 2 Bias (b^(2)):\n{model.fc2.bias.data.numpy()}")
        print(f"Final dL/dW^(1) Gradient:\n{final_grad_W1.numpy()}")
        print("=" * 60 + "\n")

    return {
        'model': model,
        'initial_loss': initial_loss.item(),
        'final_loss': final_loss,
        'early_grad_norm_w1': early_grad_norm_W1,
        'probs': final_probs,
        'preds': thresholded_preds,
        'all_correct': all_correct,
        'history': history,
        'early_grad_W1': early_grad_W1.numpy(),
        'final_grad_W1': final_grad_W1.numpy()
    }


if __name__ == '__main__':
    train_xor(activation_name='sigmoid', lr=0.05, epochs=3000, seed=42)
