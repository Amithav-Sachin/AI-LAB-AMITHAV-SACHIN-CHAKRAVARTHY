"""
Tabular Q-learning Implementation for CS F407 Reinforcement Learning Lab.
Supports:
- Training with linear epsilon decay and random tie-breaking
- Single-step Q-value update with true termination vs truncation handling
- Deterministic greedy evaluation with fixed tie-breaking
- Random policy baseline benchmark
- No-exploration ablation comparison
- Visualization and plotting of learning curves and trajectory
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt

from warehouse_env import (
    SingleRobotWarehouseEnv, UP, DOWN, LEFT, RIGHT, WAIT,
    ACTION_NAMES, pos_to_index, index_to_pos, OBSTACLES, GRID_SIZE
)


class TabularQLearningAgent:
    """Tabular Q-learning agent for 5x5 Warehouse Environment."""

    def __init__(
        self,
        num_states: int = 25,
        num_actions: int = 5,
        alpha: float = 0.2,
        gamma: float = 0.95,
        seed: int = 7
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)

    def select_action(self, state_idx: int, epsilon: float, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        If training=True, ties among max Q are broken randomly.
        If training=False (evaluation), tie is broken deterministically (first max).
        """
        if training and self.rng.rand() < epsilon:
            return self.rng.randint(self.num_actions)

        q_vals = self.q_table[state_idx]
        max_val = np.max(q_vals)
        if training:
            # Random tie-breaking among best actions
            best_actions = np.flatnonzero(np.isclose(q_vals, max_val))
            return int(self.rng.choice(best_actions))
        else:
            # Fixed deterministic tie-breaking (first index)
            return int(np.argmax(q_vals))

    def update(
        self,
        s_idx: int,
        a: int,
        r: float,
        next_s_idx: int,
        terminated: bool
    ) -> float:
        """
        One Q-learning update step:
        y = r if terminated else r + gamma * max_a' Q(s', a')
        Q(s, a) <- Q(s, a) + alpha * (y - Q(s, a))
        """
        current_q = self.q_table[s_idx, a]
        if terminated:
            target = r
        else:
            max_next_q = np.max(self.q_table[next_s_idx])
            target = r + self.gamma * max_next_q

        td_error = target - current_q
        self.q_table[s_idx, a] = current_q + self.alpha * td_error
        return td_error


def get_epsilon(episode: int, total_episodes: int = 1500, decay_episodes: int = 1200, eps_start: float = 1.0, eps_end: float = 0.05) -> float:
    """Linear epsilon decay over decay_episodes."""
    if episode >= decay_episodes:
        return eps_end
    return eps_start - (eps_start - eps_end) * (episode / decay_episodes)


def train_tabular_q(
    episodes: int = 1500,
    alpha: float = 0.2,
    gamma: float = 0.95,
    seed: int = 7,
    no_exploration: bool = False
) -> Tuple[TabularQLearningAgent, Dict[str, Any]]:
    """Train tabular Q-learning agent on SingleRobotWarehouseEnv."""
    env = SingleRobotWarehouseEnv()
    agent = TabularQLearningAgent(alpha=alpha, gamma=gamma, seed=seed)

    returns: List[float] = []
    successes: List[int] = []
    epsilons: List[float] = []
    lengths: List[int] = []
    invalid_counts: List[int] = []
    timeouts: List[int] = []

    for ep in range(episodes):
        epsilon = 0.0 if no_exploration else get_epsilon(ep, total_episodes=episodes, decay_episodes=1200)
        epsilons.append(epsilon)

        state = env.reset()
        state_idx = pos_to_index(state)
        ep_return = 0.0
        ep_invalids = 0
        done = False

        while not done:
            action = agent.select_action(state_idx, epsilon, training=True)
            next_state, r, terminated, truncated, info = env.step(action)
            next_state_idx = pos_to_index(next_state)

            if info["invalid_move"]:
                ep_invalids += 1
            ep_return += r

            # Update Q-table. Crucial: terminated suppresses continuation; truncated does NOT.
            agent.update(state_idx, action, r, next_state_idx, terminated)

            state = next_state
            state_idx = next_state_idx
            done = terminated or truncated

        returns.append(ep_return)
        successes.append(1 if terminated else 0)
        lengths.append(env.step_count)
        invalid_counts.append(ep_invalids)
        timeouts.append(1 if (truncated and not terminated) else 0)

    history = {
        "returns": np.array(returns),
        "successes": np.array(successes),
        "epsilons": np.array(epsilons),
        "lengths": np.array(lengths),
        "invalid_counts": np.array(invalid_counts),
        "timeouts": np.array(timeouts)
    }
    return agent, history


def evaluate_greedy(
    agent: TabularQLearningAgent,
    start_pos: Tuple[int, int] = (2, 0),
    max_steps: int = 40
) -> Dict[str, Any]:
    """Deterministic greedy rollout (epsilon=0, fixed tie breaking)."""
    env = SingleRobotWarehouseEnv(start=start_pos, max_steps=max_steps)
    state = env.reset(start_pos=start_pos)
    state_idx = pos_to_index(state)

    path = [state]
    actions_taken = []
    rewards = []
    invalids = 0
    done = False

    while not done:
        action = agent.select_action(state_idx, epsilon=0.0, training=False)
        actions_taken.append(action)
        next_state, r, terminated, truncated, info = env.step(action)
        path.append(next_state)
        rewards.append(r)
        if info["invalid_move"]:
            invalids += 1
        state = next_state
        state_idx = pos_to_index(state)
        done = terminated or truncated

    return {
        "start": start_pos,
        "path": path,
        "actions": actions_taken,
        "rewards": rewards,
        "total_return": sum(rewards),
        "steps": env.step_count,
        "reached_goal": path[-1] == env.goal,
        "invalid_moves": invalids,
        "timeout": (env.step_count >= max_steps) and (path[-1] != env.goal)
    }


def evaluate_random_policy(
    runs: int = 100,
    start_pos: Tuple[int, int] = (2, 0),
    seed: int = 7
) -> Dict[str, Any]:
    """Evaluate uniformly random policy across 100 runs."""
    rng = np.random.RandomState(seed)
    env = SingleRobotWarehouseEnv(start=start_pos, max_steps=40)

    success_count = 0
    timeout_count = 0
    steps_successful = []
    total_invalids = []
    returns = []

    for _ in range(runs):
        state = env.reset(start_pos=start_pos)
        done = False
        ep_invalids = 0
        ep_return = 0.0

        while not done:
            action = rng.randint(5)
            next_state, r, terminated, truncated, info = env.step(action)
            ep_return += r
            if info["invalid_move"]:
                ep_invalids += 1
            done = terminated or truncated

        returns.append(ep_return)
        total_invalids.append(ep_invalids)
        if terminated:
            success_count += 1
            steps_successful.append(env.step_count)
        else:
            timeout_count += 1

    return {
        "runs": runs,
        "success_rate": success_count / runs,
        "timeout_rate": timeout_count / runs,
        "mean_steps_successful": np.mean(steps_successful) if steps_successful else None,
        "mean_invalid_moves": np.mean(total_invalids),
        "mean_return": np.mean(returns)
    }


def render_grid_path(path: List[Tuple[int, int]], goal: Tuple[int, int] = (2, 4)) -> str:
    """ASCII grid showing the path."""
    grid = [["." for _ in range(5)] for _ in range(5)]
    for r, c in OBSTACLES:
        grid[r][c] = "#"
    grid[goal[0]][goal[1]] = "G"

    # Number the steps in path
    for t, (r, c) in enumerate(path):
        if (r, c) == path[0] and t == 0:
            grid[r][c] = "S"
        elif (r, c) == goal:
            grid[r][c] = "G"
        elif grid[r][c] == ".":
            grid[r][c] = str(t)

    lines = ["  Col: 0 1 2 3 4", "     +----------+"]
    for r in range(5):
        row_str = " ".join(grid[r])
        lines.append(f" r{r} | {row_str} |")
    lines.append("     +----------+")
    return "\n".join(lines)


def plot_tabular_learning_curves(history: Dict[str, Any], save_path: str = "tabular_q_training.png"):
    """Plot moving average return and success rate."""
    returns = history["returns"]
    successes = history["successes"]
    epsilons = history["epsilons"]
    window = 50

    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode='valid')

    ret_ma = moving_avg(returns, window)
    succ_ma = moving_avg(successes, window)
    x_vals = np.arange(window, len(returns) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(returns, alpha=0.25, color="cornflowerblue", label="Raw Episode Return")
    ax1.plot(x_vals, ret_ma, color="darkblue", lw=2, label=f"{window}-Episode Moving Average")
    ax1.set_ylabel("Return")
    ax1.set_title("Tabular Q-learning: Training Progress (Seed 7)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(x_vals, succ_ma * 100.0, color="forestgreen", lw=2, label=f"Success Rate ({window}-ep MA)")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    ax3.plot(epsilons, color="crimson", lw=2, label="Epsilon (Exploration Rate)")
    ax3.set_ylabel("Epsilon")
    ax3.set_xlabel("Episode")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved tabular training plot to {save_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(out_dir, "tabular_q_training.png")

    print("Training Tabular Q-learning Agent (1,500 episodes, seed 7)...")
    agent, history = train_tabular_q(episodes=1500, seed=7)
    plot_tabular_learning_curves(history, save_path=plot_file)

    eval_res = evaluate_greedy(agent, start_pos=(2, 0))
    print("\nGreedy Evaluation from Start (2, 0):")
    print(f"Reached goal: {eval_res['reached_goal']}")
    print(f"Steps: {eval_res['steps']}")
    print(f"Total return: {eval_res['total_return']}")
    print(f"Invalid moves: {eval_res['invalid_moves']}")
    print(f"Trajectory: {eval_res['path']}")
    print("Action sequence:", [ACTION_NAMES[a] for a in eval_res['actions']])
    print("\nASCII Grid Trajectory:")
    print(render_grid_path(eval_res['path']))

    print("\nEvaluating Uniform Random Policy (100 episodes)...")
    rand_res = evaluate_random_policy(runs=100, seed=7)
    print(f"Random Policy Success Rate: {rand_res['success_rate'] * 100:.1f}%")
    print(f"Random Policy Timeout Rate: {rand_res['timeout_rate'] * 100:.1f}%")
    print(f"Mean Steps (Successful): {rand_res['mean_steps_successful']}")
    print(f"Mean Invalid Moves: {rand_res['mean_invalid_moves']:.2f}")

    print("\nOptional Investigation: Training with No Exploration (epsilon=0)...")
    agent_no_exp, history_no_exp = train_tabular_q(episodes=1500, seed=7, no_exploration=True)
    eval_no_exp = evaluate_greedy(agent_no_exp, start_pos=(2, 0))
    print(f"No Exploration (Seed 7) Goal Reached: {eval_no_exp['reached_goal']}, Steps: {eval_no_exp['steps']}")
