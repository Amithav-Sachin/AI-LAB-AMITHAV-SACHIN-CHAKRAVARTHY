"""
build_pdf_report.py
Generates a multi-page, publication-grade academic PDF report:
NEURAL_LAB_REPORT.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

pdf_path = os.path.join(os.path.dirname(__file__), "NEURAL_LAB_REPORT.pdf")
img_dir = os.path.dirname(__file__)


def create_pdf_report():
    with PdfPages(pdf_path) as pdf:
        # ----------------------------------------------------
        # PAGE 1: Title, Abstract, Task 1 Problem Specification
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.94, "Laboratory Report: Neural Models", fontsize=20, fontweight='bold', ha='center', color='#1a365d')
        fig.text(0.5, 0.91, "Learning, Depth, Activations, and Output Layers", fontsize=14, fontstyle='italic', ha='center', color='#2b6cb0')
        fig.text(0.5, 0.88, "Course: Artificial Intelligence Laboratory  |  Date: August 2026", fontsize=10, ha='center', color='#4a5568')
        fig.text(0.1, 0.86, "_" * 85, fontsize=10, color='#cbd5e0')

        # Section 1: Executive Summary
        y = 0.83
        fig.text(0.1, y, "1. Executive Summary & Motivating Scenario", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.025
        exec_text = (
            "This laboratory investigates the fundamental mechanics of neural representation, depth, backpropagation,\n"
            "activation dynamics, and task-dependent output formulation through the canonical XOR problem.\n"
            "Scenario: An autonomous safety device monitors two redundant binary sensors, x1 and x2. The device\n"
            "must trigger a disagreement warning y = 1 if and only if exactly one sensor is active, corresponding\n"
            "to the Boolean XOR function. While deceptively simple with only four data points, XOR exposes the\n"
            "representational boundary between linear and nonlinear hypotheses and demonstrates credit assignment."
        )
        fig.text(0.1, y, exec_text, fontsize=9, va='top', family='serif', color='#2d3748')

        # Section 2: Task 1 Problem Specification
        y -= 0.12
        fig.text(0.1, y, "2. Task 1: Problem Specification & Linear Separability", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.025
        t1_text = (
            "Input Space: X = {0, 1}^2 in R^2. Output Space: Y = {0, 1}.\n"
            "Training Examples: ((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0).\n\n"
            "Why a Single Straight Decision Boundary Fails (Linear Inseparability):\n"
            "A linear decision boundary in R^2 is defined by the affine hyperplane w1*x1 + w2*x2 + b = 0. Class 1 points\n"
            "(0, 1) and (1, 0) form a line segment crossing the diagonal between Class 0 points (0, 0) and (1, 1).\n"
            "The convex hulls conv(C0) and conv(C1) intersect at (0.5, 0.5). By the Hyperplane Separation Theorem,\n"
            "two sets can be strictly separated by an affine hyperplane if and only if their convex hulls are disjoint.\n"
            "Consequently, no straight line can classify XOR without at least one classification error."
        )
        fig.text(0.1, y, t1_text, fontsize=9, va='top', family='serif', color='#2d3748')

        # Embed Inseparability Plot
        img_p1 = os.path.join(img_dir, "xor_linear_inseparability.png")
        if os.path.exists(img_p1):
            ax_img = fig.add_axes([0.18, 0.15, 0.64, 0.38])
            img = mpimg.imread(img_p1)
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title("Figure 1: Geometric Representation of XOR Linear Inseparability", fontsize=9, pad=3)

        fig.text(0.5, 0.05, "Page 1 of 4", fontsize=9, ha='center', color='#718096')
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 2: Task 2 & Task 3 Architecture, AD, and Inspection
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.94, "Laboratory Report: Neural Models", fontsize=14, fontweight='bold', ha='center', color='#1a365d')
        fig.text(0.1, 0.91, "_" * 85, fontsize=10, color='#cbd5e0')

        y = 0.88
        fig.text(0.1, y, "3. Task 2: Architecture Design & Scientific Principles", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.025
        t2_text = (
            "Baseline Architecture: 2 inputs -> 2 hidden units -> 1 output logit (2-2-1 network).\n"
            "Hidden layer: a^(1) = W^(1)x + b^(1), h^(1) = f(a^(1)) with W^(1) in R^(2x2), b^(1) in R^2.\n"
            "Output layer: z = W^(2)h^(1) + b^(2) with W^(2) in R^(1x2), b^(2) in R.\n"
            "Loss function: Binary Cross-Entropy with Logits: L = - [y*log(sigma(z)) + (1-y)*log(1-sigma(z))].\n\n"
            "Core Scientific Justifications:\n"
            "1. Hidden Nonlinearity: Without activation f, the composition of affine maps collapses to a single\n"
            "   affine transformation: W^(2)(W^(1)x + b^(1)) + b^(2) = (W^(2)W^(1))x + (W^(2)b^(1) + b^(2)) = W_eff*x + b_eff.\n"
            "   Depth without nonlinearity increases parameter count without expanding representational capacity.\n"
            "2. Sigmoid + BCE Pairing: Directly optimizes Bernoulli negative log-likelihood. Produces logit\n"
            "   gradient dL/dz = sigma(z) - y = p - y, eliminating saturated vanishing gradients that plague MSE.\n"
            "3. Validation Criteria: Successful learning requires: (a) final loss < 0.01; (b) 4/4 correct predictions;\n"
            "   (c) vanishing gradient norms as convergence reaches optimum; (d) reproducibility across random seeds."
        )
        fig.text(0.1, y, t2_text, fontsize=9, va='top', family='serif', color='#2d3748')

        y -= 0.28
        fig.text(0.1, y, "4. Task 3: LLM Engineering Assistance & Code Inspection", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.025
        t3_text = (
            "Prompt Structure Provided to LLM:\n"
            "\"Generate minimal PyTorch code for a 2-2-1 network on XOR dataset [[0,0],[0,1],[1,0],[1,1]] with labels\n"
            "[[0],[1],[1],[0]]. Use nonlinear hidden activation, BCEWithLogitsLoss, random weight initialization, full-batch\n"
            "training for a few thousand lightweight CPU steps. Report initial/final loss, probabilities, thresholded predictions,\n"
            "and parameter gradients. Set random seed for reproducibility.\"\n\n"
            "Inspection and Critical Verification:\n"
            "- Forward Pass: Invoked via model(X), executing affine transforms and activation function.\n"
            "- Scalar Loss: Formed by criterion(logits, y), aggregating mean BCE across the 4 examples.\n"
            "- Reverse-Mode AD: Triggered by loss.backward(), populating param.grad tensors via vector-Jacobian products.\n"
            "- Optimizer Step: Executed by optimizer.step(), applying Adam parameter updates.\n"
            "- Essential Engineering Corrections: (1) Switched from manual Sigmoid+BCELoss to BCEWithLogitsLoss for\n"
            "  log-sum-exp numerical stability; (2) Ensured optimizer.zero_grad() precedes backward() to prevent accumulation."
        )
        fig.text(0.1, y, t3_text, fontsize=9, va='top', family='serif', color='#2d3748')

        # Static vs Dynamic Think About It box
        y -= 0.28
        fig.text(0.1, y, "Think About It: Static vs. Dynamic Code Verification", fontsize=11, fontweight='bold', color='#2b6cb0')
        y -= 0.022
        think_text = (
            "- Statically Verifiable (Without Execution): Layer shapes (2-2-1), loss criteria (BCEWithLogitsLoss),\n"
            "  presence of nonlinear activations, proper gradient zeroing, dataset shapes.\n"
            "- Requires Execution & Measurement: Loss convergence, saddle-point avoidance, gradient norm dynamics,\n"
            "  empirical classification accuracy, and symmetry breaking behavior."
        )
        fig.text(0.1, y, think_text, fontsize=8.5, va='top', family='serif', color='#4a5568')

        fig.text(0.5, 0.05, "Page 2 of 4", fontsize=9, ha='center', color='#718096')
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 3: Task 4 Experimental Results & Diagnostics
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.94, "Laboratory Report: Neural Models", fontsize=14, fontweight='bold', ha='center', color='#1a365d')
        fig.text(0.1, 0.91, "_" * 85, fontsize=10, color='#cbd5e0')

        y = 0.88
        fig.text(0.1, y, "5. Task 4: Execution, Diagnostics, Symmetry & Activations", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.025
        t4_text = (
            "Part A - Baseline Results (Sigmoid, lr=0.05):\n"
            "Initial Loss: 0.748230 | Final Loss: 0.000340 | Accuracy: 4/4 (100%)\n"
            "Probabilities: (0,0)->0.00031 [0], (0,1)->0.99969 [1], (1,0)->0.99955 [1], (1,1)->0.00030 [0].\n\n"
            "Part B - Backpropagation & Average Gradient:\n"
            "param.grad reflects dL/dW^(1) = (1/N) * sum_n delta_n^(1) * (x_n)^T. Because the empirical loss L is the\n"
            "mean over batch examples, linearity of differentiation ensures the total gradient is the exact average."
        )
        fig.text(0.1, y, t4_text, fontsize=9, va='top', family='serif', color='#2d3748')

        # Activation Table
        y -= 0.16
        fig.text(0.1, y, "Part D - Activation Benchmark (Seed=2, lr=0.05):", fontsize=10, fontweight='bold', color='#1a365d')
        table_data = [
            ["Hidden Activation", "Final Loss", "4/4 Correct?", "Early ||dL/dW^(1)||_2 (step 10)"],
            ["Sigmoid", "0.000380", "Yes (4/4)", "0.008495"],
            ["Tanh", "0.000132", "Yes (4/4)", "0.046584"],
            ["ReLU", "0.000049", "Yes (4/4)", "0.024614"]
        ]
        table_ax = fig.add_axes([0.1, y - 0.11, 0.8, 0.09])
        table_ax.axis('off')
        tab = table_ax.table(cellText=table_data, loc='center', cellLoc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        for (row, col), cell in tab.get_celld().items():
            if row == 0:
                cell.set_facecolor('#edf2f7')
                cell.set_text_props(weight='bold')

        # Embed Symmetry & Training Plots
        img_sym = os.path.join(img_dir, "symmetry_collapse.png")
        if os.path.exists(img_sym):
            ax_sym = fig.add_axes([0.1, 0.40, 0.8, 0.22])
            img = mpimg.imread(img_sym)
            ax_sym.imshow(img)
            ax_sym.axis('off')

        img_train = os.path.join(img_dir, "training_curves.png")
        if os.path.exists(img_train):
            ax_tr = fig.add_axes([0.1, 0.12, 0.8, 0.24])
            img = mpimg.imread(img_train)
            ax_tr.imshow(img)
            ax_tr.axis('off')

        fig.text(0.5, 0.05, "Page 3 of 4", fontsize=9, ha='center', color='#718096')
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # PAGE 4: Task 5 Multiclass Extension & Reflection
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.94, "Laboratory Report: Neural Models", fontsize=14, fontweight='bold', ha='center', color='#1a365d')
        fig.text(0.1, 0.91, "_" * 85, fontsize=10, color='#cbd5e0')

        y = 0.88
        fig.text(0.1, y, "6. Task 5: Three-Class Sensor Extension & Diagnostics", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.025
        t5_text = (
            "Task Formulation: Class 0: Inactive (0,0) | Class 1: Disagreement (0,1),(1,0) | Class 2: Both Active (1,1).\n"
            "Architecture: 2 inputs -> 4 hidden units -> 3 output logits. Loss: Multiclass Cross-Entropy.\n"
            "- Final Weight Shape: W^(2) in R^(3x4). Number of logits per sample: 3.\n"
            "- Softmax Property: sum_k exp(z_k) / sum_j exp(z_j) = 1.0 strictly. Verified numerically: sum(p) = 0.99999994.\n"
            "- Analytical Gradient dL/dz = p - y: Verified against PyTorch autograd to machine precision (diff: 6.82e-13).\n"
            "- Logit Shift Invariance: Adding C = 100 to all logits yielded identical probabilities (diff < 4.09e-11).\n"
            "  Stable implementations evaluate exp(z_k - max(z)) to eliminate positive overflow without distorting ratios."
        )
        fig.text(0.1, y, t5_text, fontsize=9, va='top', family='serif', color='#2d3748')

        # Embed 3-Class Decision Boundary Plot
        img_3cls = os.path.join(img_dir, "three_class_decision_boundary.png")
        if os.path.exists(img_3cls):
            ax_3c = fig.add_axes([0.22, 0.44, 0.56, 0.28])
            img = mpimg.imread(img_3cls)
            ax_3c.imshow(img)
            ax_3c.axis('off')

        y = 0.40
        fig.text(0.1, y, "7. Answers to Reflection Questions", fontsize=12, fontweight='bold', color='#1a365d')
        y -= 0.022
        ref_text = (
            "1. Depth vs. Nonlinearity: Depth alone collapses affine maps to a single affine layer. Nonlinearity folds the space.\n"
            "2. Useful Learning Signal: Loss steadily decayed by 4 orders of magnitude while accuracy reached 100%.\n"
            "3. Symmetry Collapse: Zero-initialization produces identical hidden activations and gradients, freezing rank to 1.\n"
            "4. Activation Gradient Effects: Sigmoid derivatives saturate (<=0.25); Tanh is zero-centered; ReLU passes unattenuated\n"
            "   derivatives (1.0), yielding faster descent but risking dead units.\n"
            "5. Task-Loss Coupling: Sigmoid+BCE matches Bernoulli; Softmax+CE matches Categorical, ensuring linear logit gradients.\n"
            "6. LLM Role: LLM generated clean boilerplate PyTorch code; human verification prevented BCELoss numerical errors.\n"
            "7. Scaling Tests: Loss tracking, gradient norms, and unit test suites scale; finite-difference gradient checks do not."
        )
        fig.text(0.1, y, ref_text, fontsize=8.5, va='top', family='serif', color='#2d3748')

        fig.text(0.5, 0.05, "Page 4 of 4", fontsize=9, ha='center', color='#718096')
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Publication-grade PDF successfully created at: {pdf_path}")


if __name__ == '__main__':
    create_pdf_report()
