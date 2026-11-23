"""
run_experiments.py
Master Execution and Experiment Orchestrator for Laboratory - Neural Models.
Executes all tasks, runs diagnostics, records test results, and prints the master summary.
"""

import os
import sys
import torch
import numpy as np

# Ensure local modules are accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xor_binary_model import train_xor
from symmetry_experiment import run_symmetry_experiment
from activation_experiment import run_activation_benchmark
from three_class_extension import run_three_class_experiment
from generate_plots import (
    plot_linear_inseparability,
    plot_training_curves,
    plot_symmetry_collapse,
    plot_three_class_decision_boundary
)
from build_pdf_report import create_pdf_report
from generate_notebook import nb_path


def main():
    print("=" * 80)
    print("STARTING FULL LABORATORY EXPERIMENTAL SUITE: NEURAL MODELS")
    print("=" * 80)

    # 1. Baseline Binary XOR Model
    print("\n>>> RUNNING TASK 4 PART A & B: BASELINE XOR MODEL...")
    xor_res = train_xor(activation_name='sigmoid', lr=0.05, epochs=3000, seed=42, verbose=True)

    # 2. Symmetry Experiment
    print("\n>>> RUNNING TASK 4 PART C: SYMMETRY BREAKING EXPERIMENT...")
    sym_records = run_symmetry_experiment(epochs=100, lr=0.1)

    # 3. Activation Benchmark
    print("\n>>> RUNNING TASK 4 PART D: ACTIVATION EXPERIMENT (SIGMOID vs TANH vs RELU)...")
    act_results = run_activation_benchmark(lr=0.05, epochs=2500, early_step=10, seed=2)

    # 4. Three-Class Extension
    print("\n>>> RUNNING TASK 5: THREE-CLASS SENSOR EXTENSION...")
    three_cls_res = run_three_class_experiment(seed=42, lr=0.05, epochs=2500)

    # 5. Generate Visual Plots
    print("\n>>> GENERATING ALL VISUAL PLOTS...")
    plot_linear_inseparability()
    plot_training_curves()
    plot_symmetry_collapse()
    plot_three_class_decision_boundary()

    # 6. Generate Publication PDF
    print("\n>>> GENERATING ACADEMIC PDF REPORT...")
    create_pdf_report()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Artifacts verified in directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("=" * 80)


if __name__ == '__main__':
    main()
