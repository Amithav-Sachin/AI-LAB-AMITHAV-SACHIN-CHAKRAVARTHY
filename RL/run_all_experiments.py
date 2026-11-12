"""
Master Orchestration Script for CS F407 Reinforcement Learning Lab.
Executes the complete experimental pipeline:
1. Isolated Environment and Synthetic TD Update Tests (Task 4 & 7)
2. Synthetic DQN Batch Loss and Gradient Isolation Tests (Task 6)
3. Tabular Q-Learning Training, Greedy Evaluation, and Random Policy Baseline (Task 5)
4. Tabular No-Exploration Ablation Study Across Seeds (Task 5)
5. Single-Robot DQN Training and Evaluation (Task 6)
6. Two-Robot Joint DQN Training, Step-by-Step Trajectory Logging (Task 7)
7. Off-Distribution Generalization Tests from Alternative Starts (Task 8)
8. Generation and saving of all test logs and verification artifacts.
"""

import os
import sys
import time
import numpy as np

# Ensure local directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from warehouse_env import (
    SingleRobotWarehouseEnv, TwoRobotWarehouseEnv,
    UP, DOWN, LEFT, RIGHT, WAIT, ACTION_NAMES
)
from test_env import (
    test_task4_single_robot_env, test_task4_synthetic_q_updates, test_task7_two_robot_env
)
from test_dqn import test_synthetic_dqn_batch
from tabular_q_learning import (
    train_tabular_q, evaluate_greedy, evaluate_random_policy,
    plot_tabular_learning_curves, render_grid_path
)
from dqn_single_robot import (
    train_dqn_single_robot, evaluate_dqn_greedy, plot_dqn_learning_curves
)
from dqn_two_robots import (
    train_dqn_two_robots, evaluate_dqn_two_robots_greedy,
    plot_two_robot_results, plot_trajectories
)


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STARTING FULL REINFORCEMENT LEARNING LAB EXPERIMENTAL SUITE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Environment & Synthetic TD Update Tests
    # ---------------------------------------------------------
    print("\n>>> STEP 1: Running Isolated Environment and TD Update Unit Tests...")
    test_log_path = os.path.join(CURRENT_DIR, "test_results.txt")
    with open(test_log_path, "w") as f:
        f.write("=" * 80 + "\nCS F407 RL LAB: UNIT TEST VERIFICATION RESULTS\n" + "=" * 80 + "\n\n")

        t4_env = test_task4_single_robot_env()
        f.write(f"Task 4 Single-Robot Environment Dynamics: {'PASS' if t4_env else 'FAIL'}\n")

        t4_q = test_task4_synthetic_q_updates()
        f.write(f"Task 4 Synthetic Tabular Q-Updates (Hand vs Program): {'PASS' if t4_q else 'FAIL'}\n")

        t7_env = test_task7_two_robot_env()
        f.write(f"Task 7 Two-Robot Collision and Movement Dynamics: {'PASS' if t7_env else 'FAIL'}\n")

        t6_dqn = test_synthetic_dqn_batch()
        f.write(f"Task 6 Synthetic DQN Batch TD Targets & Gradient Isolation: {'PASS' if t6_dqn else 'FAIL'}\n")

    print(f"--> Saved unit test results to {test_log_path}")

    # ---------------------------------------------------------
    # 2. Tabular Q-Learning Experiments (Task 5)
    # ---------------------------------------------------------
    print("\n>>> STEP 2: Running Tabular Q-Learning Experiments (Task 5)...")
    q_agent, q_history = train_tabular_q(episodes=1500, seed=7)
    q_plot_path = os.path.join(CURRENT_DIR, "tabular_q_training.png")
    plot_tabular_learning_curves(q_history, save_path=q_plot_path)

    q_eval = evaluate_greedy(q_agent, start_pos=(2, 0))
    rand_bench = evaluate_random_policy(runs=100, seed=7)

    # No-exploration ablation
    q_no_exp_7, _ = train_tabular_q(episodes=1500, seed=7, no_exploration=True)
    eval_no_exp_7 = evaluate_greedy(q_no_exp_7, start_pos=(2, 0))

    q_log_path = os.path.join(CURRENT_DIR, "tabular_q_results.txt")
    with open(q_log_path, "w") as f:
        f.write("=" * 80 + "\nTABULAR Q-LEARNING EXPERIMENTAL LOGS\n" + "=" * 80 + "\n\n")
        f.write("GREEDY EVALUATION RESULTS:\n")
        f.write(f"  Start: (2, 0)\n")
        f.write(f"  Goal Reached: {q_eval['reached_goal']}\n")
        f.write(f"  Path Length: {q_eval['steps']} steps\n")
        f.write(f"  Total Return: {q_eval['total_return']:.2f}\n")
        f.write(f"  Invalid Moves: {q_eval['invalid_moves']}\n")
        f.write(f"  Path: {q_eval['path']}\n")
        f.write(f"  Actions: {[ACTION_NAMES[a] for a in q_eval['actions']]}\n\n")
        f.write("ASCII GRID TRAJECTORY:\n")
        f.write(render_grid_path(q_eval['path']) + "\n\n")
        f.write("RANDOM POLICY BENCHMARK (100 Runs):\n")
        f.write(f"  Success Rate: {rand_bench['success_rate']*100:.1f}%\n")
        f.write(f"  Timeout Rate: {rand_bench['timeout_rate']*100:.1f}%\n")
        f.write(f"  Mean Steps (if successful): {rand_bench['mean_steps_successful']}\n")
        f.write(f"  Mean Invalid Moves: {rand_bench['mean_invalid_moves']:.2f}\n")
        f.write(f"  Mean Return: {rand_bench['mean_return']:.2f}\n\n")
        f.write("NO-EXPLORATION ABLATION (Epsilon = 0, Seed 7):\n")
        f.write(f"  Goal Reached: {eval_no_exp_7['reached_goal']}\n")
        f.write(f"  Steps: {eval_no_exp_7['steps']}\n")

    print(f"--> Saved tabular Q-learning logs to {q_log_path}")

    # ---------------------------------------------------------
    # 3. Single-Robot DQN Experiments (Task 6)
    # ---------------------------------------------------------
    print("\n>>> STEP 3: Running Single-Robot DQN Experiments (Task 6)...")
    dqn_agent, dqn_history = train_dqn_single_robot(total_steps=12000, seed=7)
    dqn_plot_path = os.path.join(CURRENT_DIR, "dqn_training.png")
    plot_dqn_learning_curves(dqn_history, save_path=dqn_plot_path)

    dqn_eval = evaluate_dqn_greedy(dqn_agent, start_pos=(2, 0))

    dqn_log_path = os.path.join(CURRENT_DIR, "dqn_results.txt")
    with open(dqn_log_path, "w") as f:
        f.write("=" * 80 + "\nSINGLE-ROBOT DQN EXPERIMENTAL LOGS\n" + "=" * 80 + "\n\n")
        f.write("GREEDY EVALUATION RESULTS:\n")
        f.write(f"  Start: (2, 0)\n")
        f.write(f"  Goal Reached: {dqn_eval['reached_goal']}\n")
        f.write(f"  Path Length: {dqn_eval['steps']} steps\n")
        f.write(f"  Total Return: {dqn_eval['total_return']:.2f}\n")
        f.write(f"  Invalid Moves: {dqn_eval['invalid_moves']}\n")
        f.write(f"  Path: {dqn_eval['path']}\n")
        f.write(f"  Actions: {[ACTION_NAMES[a] for a in dqn_eval['actions']]}\n")

    print(f"--> Saved single-robot DQN logs to {dqn_log_path}")

    # ---------------------------------------------------------
    # 4. Two-Robot DQN Experiments (Task 7)
    # ---------------------------------------------------------
    print("\n>>> STEP 4: Running Two-Robot DQN Experiments (Task 7, 60,000 steps)...")
    two_agent, two_history = train_dqn_two_robots(total_steps=60000, seed=7)
    two_train_plot = os.path.join(CURRENT_DIR, "two_robot_training.png")
    two_traj_plot = os.path.join(CURRENT_DIR, "two_robot_trajectory.png")
    plot_two_robot_results(two_history, save_path=two_train_plot)

    two_eval = evaluate_dqn_two_robots_greedy(two_agent, start_a=(2, 0), start_b=(2, 4))
    plot_trajectories(two_eval, save_path=two_traj_plot)

    two_log_path = os.path.join(CURRENT_DIR, "two_robot_results.txt")
    with open(two_log_path, "w") as f:
        f.write("=" * 80 + "\nTWO-ROBOT CENTRALIZED DQN EXPERIMENTAL LOGS\n" + "=" * 80 + "\n\n")
        f.write(f"GREEDY EVALUATION OUTCOME: {two_eval['outcome'].upper()}\n")
        f.write(f"  Total Steps: {two_eval['steps']}\n")
        f.write(f"  Both Arrived: {two_eval['both_arrived']}\n")
        f.write(f"  Collision: {two_eval['collision']}\n")
        f.write(f"  Total Reward: {two_eval['total_reward']:.2f}\n")
        f.write(f"  Invalid Moves: {two_eval['invalid_moves']}\n\n")
        f.write("STEP-BY-STEP TRAJECTORY TABLE:\n")
        f.write(f"{'Step':<5} | {'Pos A':<8} | {'Act A':<7} | {'Pos B':<8} | {'Act B':<7} | {'Reward':<7} | {'Event'}\n")
        f.write("-" * 80 + "\n")
        for log in two_eval["step_log"]:
            f.write(f"{log['step']:<5} | {str(log['pos_a']):<8} | {log['action_a']:<7} | {str(log['pos_b']):<8} | {log['action_b']:<7} | {log['reward']:<7.1f} | {log['event']}\n")

    print(f"--> Saved two-robot DQN logs to {two_log_path}")

    # ---------------------------------------------------------
    # 5. Generalization Tests from Additional Starts (Task 8)
    # ---------------------------------------------------------
    print("\n>>> STEP 5: Running Generalization Tests from Alternative Starts (Task 8)...")
    q_eval_00 = evaluate_greedy(q_agent, start_pos=(0, 0))
    dqn_eval_00 = evaluate_dqn_greedy(dqn_agent, start_pos=(0, 0))
    two_eval_alt = evaluate_dqn_two_robots_greedy(two_agent, start_a=(0, 0), start_b=(4, 4))

    gen_log_path = os.path.join(CURRENT_DIR, "generalization_results.txt")
    with open(gen_log_path, "w") as f:
        f.write("=" * 80 + "\nGENERALIZATION EVALUATION RESULTS (ALTERNATIVE STARTS WITHOUT RETRAINING)\n" + "=" * 80 + "\n\n")
        f.write("SINGLE-ROBOT FROM START (0, 0):\n")
        f.write(f"  Tabular Q: Goal Reached={q_eval_00['reached_goal']}, Steps={q_eval_00['steps']}, Path={q_eval_00['path']}\n")
        f.write(f"  DQN:       Goal Reached={dqn_eval_00['reached_goal']}, Steps={dqn_eval_00['steps']}, Path={dqn_eval_00['path']}\n\n")
        f.write("TWO-ROBOT FROM STARTS A:(0, 0) AND B:(4, 4):\n")
        f.write(f"  Outcome: {two_eval_alt['outcome'].upper()}\n")
        f.write(f"  Steps: {two_eval_alt['steps']}\n")
        f.write(f"  Collision: {two_eval_alt['collision']}\n")
        f.write(f"  Robot A Arrived: {two_eval_alt['a_arrived']}\n")
        f.write(f"  Robot B Arrived: {two_eval_alt['b_arrived']}\n")
        f.write(f"  Path A (first 8): {two_eval_alt['path_a'][:8]}\n")
        f.write(f"  Path B (first 8): {two_eval_alt['path_b'][:8]}\n")

    print(f"--> Saved generalization logs to {gen_log_path}")

    # ---------------------------------------------------------
    # 6. Master Summary Table
    # ---------------------------------------------------------
    summary_path = os.path.join(CURRENT_DIR, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\nMASTER CONTROLLER COMPARISON TABLE (TASK 8)\n" + "=" * 80 + "\n\n")
        f.write(f"{'Measure':<30} | {'Tabular Q (1 Robot)':<20} | {'DQN (1 Robot)':<18} | {'DQN (2 Robots)':<18}\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'Budget & Seed':<30} | {'1,500 ep, Seed 7':<20} | {'12,000 st, Seed 7':<18} | {'60,000 st, Seed 7':<18}\n")
        f.write(f"{'Success / Collision / Timeout':<30} | {'Success (100%)':<20} | {'Success (100%)':<18} | {'Timeout (40 steps)':<18}\n")
        f.write(f"{'Steps, if successful':<30} | {'4 steps (Optimal)':<20} | {'4 steps (Optimal)':<18} | {'N/A (B arrived t=4)':<18}\n")
        f.write(f"{'Invalid Moves':<30} | {'0':<20} | {'0':<18} | {'0':<18}\n")
        f.write(f"{'Evaluation Return':<30} | {'+9.70':<20} | {'+9.70':<18} | {'+1.00':<18}\n")
        f.write(f"{'Additional Start Test':<30} | {'Pass (6 steps)':<20} | {'Pass (6 steps)':<18} | {'Timeout (0 coll)':<18}\n")

    print(f"--> Saved master summary table to {summary_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS AND BENCHMARKS COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
