"""
generate_plots.py
Generates visual diagrams and plots for Laboratory - Neural Models.
Outputs:
  - xor_linear_inseparability.png
  - training_curves.png
  - symmetry_collapse.png
  - three_class_decision_boundary.png
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from xor_binary_model import train_xor
from symmetry_experiment import XORNetZeroInit
from activation_experiment import train_single_activation
from three_class_extension import ThreeClassSensorNet

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dir = os.path.dirname(os.path.abspath(__file__))


def plot_linear_inseparability():
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Plot points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='#1f77b4', s=250, label='Class 0: Agreement (y=0)', zorder=5, edgecolors='black', linewidth=1.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='#d62728', s=250, marker='s', label='Class 1: Disagreement (y=1)', zorder=5, edgecolors='black', linewidth=1.5)

    # Annotate points
    labels = ["(0,0)\ny=0", "(0,1)\ny=1", "(1,0)\ny=1", "(1,1)\ny=0"]
    offsets = [(-0.12, -0.12), (-0.12, 0.08), (0.05, -0.12), (0.05, 0.08)]
    for i, txt in enumerate(labels):
        ax.annotate(txt, (X[i, 0], X[i, 1]), xytext=(X[i, 0] + offsets[i][0], X[i, 1] + offsets[i][1]),
                    fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Draw convex hulls / crossing diagonals
    ax.plot([0, 1], [0, 1], 'b--', alpha=0.5, linewidth=2, label='Convex Hull Class 0')
    ax.plot([0, 1], [1, 0], 'r--', alpha=0.5, linewidth=2, label='Convex Hull Class 1')
    ax.scatter([0.5], [0.5], color='purple', s=120, marker='X', zorder=6, label='Intersection (0.5, 0.5)')

    # Attempted linear boundaries
    x_vals = np.linspace(-0.3, 1.3, 100)
    ax.plot(x_vals, 0.5 - x_vals, 'k:', alpha=0.6, label='Candidate Linear Boundary (Fails)')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Sensor $x_1$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensor $x_2$', fontsize=12, fontweight='bold')
    ax.set_title('Task 1: XOR Linear Inseparability in $\\mathcal{X} = \\{0, 1\\}^2$', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(fig_dir, 'xor_linear_inseparability.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_training_curves():
    epochs = 2500
    res_sig = train_single_activation('sigmoid', lr=0.05, epochs=epochs, seed=2)
    res_tanh = train_single_activation('tanh', lr=0.05, epochs=epochs, seed=2)
    res_relu = train_single_activation('relu', lr=0.05, epochs=epochs, seed=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves
    ax1.plot(res_sig['loss_history'], label='Sigmoid', color='#1f77b4', linewidth=2)
    ax1.plot(res_tanh['loss_history'], label='Tanh', color='#2ca02c', linewidth=2)
    ax1.plot(res_relu['loss_history'], label='ReLU', color='#d62728', linewidth=2)
    ax1.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('BCE With Logits Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Task 4 Part D: Loss Convergence Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')

    # Gradient Norm Curves
    ax2.plot(res_sig['grad_norm_history'], label='Sigmoid', color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax2.plot(res_tanh['grad_norm_history'], label='Tanh', color='#2ca02c', linewidth=1.5, alpha=0.8)
    ax2.plot(res_relu['grad_norm_history'], label='ReLU', color='#d62728', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('$\\|\\nabla_{W^{(1)}} L\\|_2$ Gradient Norm', fontsize=11, fontweight='bold')
    ax2.set_title('Gradient Norm Evolution Across Activations', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(fig_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_symmetry_collapse():
    # Simulate step-by-step weight changes under zero init
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    model = XORNetZeroInit(activation_fn=nn.Sigmoid)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    row1_w11, row1_w12 = [], []
    row2_w21, row2_w22 = [], []
    losses = []

    for step in range(150):
        w1 = model.fc1.weight.detach().numpy()
        row1_w11.append(w1[0, 0])
        row1_w12.append(w1[0, 1])
        row2_w21.append(w1[1, 0])
        row2_w22.append(w1[1, 1])

        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Weight Trajectories
    steps = range(150)
    ax1.plot(steps, row1_w11, 'b-', label='Neuron 1: $W_{11}^{(1)}$', linewidth=2)
    ax1.plot(steps, row2_w21, 'r--', label='Neuron 2: $W_{21}^{(1)}$', linewidth=2)
    ax1.plot(steps, row1_w12, 'c-.', label='Neuron 1: $W_{12}^{(1)}$', linewidth=2)
    ax1.plot(steps, row2_w22, 'm:', label='Neuron 2: $W_{22}^{(1)}$', linewidth=2)
    ax1.set_xlabel('Optimization Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Weight Value', fontsize=11, fontweight='bold')
    ax1.set_title('Task 4 Part C: Hidden Weight Symmetry ($W_1 = W_2$ Forever)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    # Loss Stalling
    ax2.plot(steps, losses, color='#d62728', linewidth=2.5)
    ax2.set_xlabel('Optimization Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('BCE Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Loss Trapped at $\\ln(2) \\approx 0.69315$ (No Learning)', fontsize=12, fontweight='bold')
    ax2.axhline(0.693147, color='black', linestyle='--', label='$\\ln(2) = 0.693147$')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(fig_dir, 'symmetry_collapse.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_three_class_decision_boundary():
    # Train 3-class model
    torch.manual_seed(42)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    model = ThreeClassSensorNet(d_hidden=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(2500):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # Generate 2D meshgrid
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200), np.linspace(-0.2, 1.2, 200))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits_grid, _ = model(grid_tensor)
        probs_grid = torch.softmax(logits_grid, dim=1).numpy()
        pred_grid = np.argmax(probs_grid, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.contourf(xx, yy, pred_grid, alpha=0.35, levels=[-0.5, 0.5, 1.5, 2.5], colors=['#4575b4', '#ffffbf', '#d73027'])

    # Scatter data points
    colors = ['blue', 'goldenrod', 'red']
    markers = ['o', 's', '^']
    labels = ['Class 0: Both Inactive (0,0)', 'Class 1: Disagree (0,1),(1,0)', 'Class 2: Both Active (1,1)']

    for cls in range(3):
        idx = (y.numpy() == cls)
        ax.scatter(X[idx, 0], X[idx, 1], color=colors[cls], marker=markers[cls], s=200, edgecolors='black', linewidth=1.5, label=labels[cls], zorder=5)

    ax.set_xlabel('Sensor $x_1$', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sensor $x_2$', fontsize=11, fontweight='bold')
    ax.set_title('Task 5: Three-Class Sensor Partitioning Surface', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(fig_dir, 'three_class_decision_boundary.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    print("Generating all visualization plots...")
    plot_linear_inseparability()
    plot_training_curves()
    plot_symmetry_collapse()
    plot_three_class_decision_boundary()
    print("All plots generated successfully!")
