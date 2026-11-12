"""
Deep Q-Network (DQN) for Two-Robot Warehouse Coordination and Collision Avoidance.
Fulfills all requirements of Task 7:
- Joint state: 50 inputs (concatenation of length-25 one-hot vectors for A and B)
- Joint action: 25 linear outputs (5x5 joint actions, decoded via divmod(j, 5))
- Replay capacity: 20,000, batch size: 64
- Collect 500 warmup steps before learning
- 1 optimizer update every 4 environment steps
- Target network updated every 250 optimizer updates
- Adam optimizer (lr=0.001), MSE TD error loss
- 60,000 environment steps, seed 7
- Epsilon linear decay: 1.0 -> 0.05 over first 80% (48,000 steps)
- Detailed step-by-step table and trajectory plots
"""

import os
import random
from collections import deque
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warehouse_env import (
    TwoRobotWarehouseEnv, UP, DOWN, LEFT, RIGHT, WAIT,
    ACTION_NAMES, pos_to_index, index_to_pos, pos_to_one_hot,
    OBSTACLES, GRID_SIZE
)


def joint_state_to_features(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> np.ndarray:
    """Concatenate two length-25 one-hot vectors to form a length-50 vector."""
    vec_a = pos_to_one_hot(pos_a, size=25)
    vec_b = pos_to_one_hot(pos_b, size=25)
    return np.concatenate([vec_a, vec_b], axis=0)


class TwoRobotQNetwork(nn.Module):
    """Feedforward Q-Network: 50 -> 64 -> 64 -> 25."""

    def __init__(self, in_features: int = 50, hidden_dim: int = 64, num_actions: int = 25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoRobotReplayBuffer:
    """Replay buffer for two-robot transitions."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_feat: np.ndarray, action: int, reward: float, next_state_feat: np.ndarray, done: bool):
        self.buffer.append((state_feat, action, reward, next_state_feat, done))

    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class TwoRobotDQNAgent:
    """Centralized Joint DQN Agent for Two Robots."""

    def __init__(
        self,
        in_features: int = 50,
        num_actions: int = 25,
        lr: float = 0.001,
        gamma: float = 0.95,
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        target_update_freq: int = 250,
        seed: int = 7
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.online_net = TwoRobotQNetwork(in_features, 64, num_actions)
        self.target_net = TwoRobotQNetwork(in_features, 64, num_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = TwoRobotReplayBuffer(capacity=buffer_capacity)

        self.optimizer_updates = 0

    def select_action(self, state_feat: np.ndarray, epsilon: float, training: bool = True) -> int:
        """Epsilon-greedy joint action selection."""
        if training and random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            s_t = torch.tensor(state_feat, dtype=torch.float32).unsqueeze(0)
            q_vals = self.online_net(s_t).squeeze(0).numpy()
            return int(np.argmax(q_vals))

    def update_step(self) -> Optional[float]:
        """Perform one gradient update on batch of 64 transitions."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 1. Online Q-values for recorded joint actions
        q_all = self.online_net(states)
        q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Target Q-values: detached with torch.no_grad()
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            max_next_q, _ = next_q_all.max(dim=1)
            # Terminal states (collision or both at goals) suppress continuation
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.optimizer_updates += 1
        if self.optimizer_updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()


def get_two_robot_epsilon(step: int, total_steps: int = 60000, decay_fraction: float = 0.80, eps_start: float = 1.0, eps_end: float = 0.05) -> float:
    """Linear epsilon decay over first 80% (48,000 steps)."""
    decay_steps = int(total_steps * decay_fraction)
    if step >= decay_steps:
        return eps_end
    return eps_start - (eps_start - eps_end) * (step / decay_steps)


def train_dqn_two_robots(
    total_steps: int = 60000,
    warmup_steps: int = 500,
    update_interval: int = 4,
    target_update_freq: int = 250,
    lr: float = 0.001,
    seed: int = 7
) -> Tuple[TwoRobotDQNAgent, Dict[str, Any]]:
    """Train Two-Robot DQN joint controller for 60,000 environment steps."""
    env = TwoRobotWarehouseEnv()
    agent = TwoRobotDQNAgent(
        in_features=50,
        num_actions=25,
        lr=lr,
        gamma=0.95,
        buffer_capacity=20000,
        batch_size=64,
        target_update_freq=target_update_freq,
        seed=seed
    )

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_success: List[int] = []
    episode_collisions: List[int] = []
    episode_timeouts: List[int] = []
    episode_invalids: List[int] = []
    loss_history: List[float] = []

    pos_a, pos_b = env.reset()
    state_feat = joint_state_to_features(pos_a, pos_b)
    ep_ret = 0.0
    ep_len = 0
    ep_inv = 0

    for step in range(1, total_steps + 1):
        eps = get_two_robot_epsilon(step, total_steps=total_steps, decay_fraction=0.80)
        joint_action = agent.select_action(state_feat, eps, training=True)

        (next_pos_a, next_pos_b), reward, terminated, truncated, info = env.step(joint_action)
        next_state_feat = joint_state_to_features(next_pos_a, next_pos_b)

        ep_ret += reward
        ep_len += 1
        ep_inv += info["invalid_moves"]

        # Push to replay: 'terminated' is true on collision or both arrived
        agent.memory.push(state_feat, joint_action, reward, next_state_feat, terminated)

        state_feat = next_state_feat
        pos_a, pos_b = next_pos_a, next_pos_b

        # Update weights every 4 steps after warmup
        if step >= warmup_steps and step % update_interval == 0:
            loss = agent.update_step()
            if loss is not None:
                loss_history.append(loss)

        if terminated or truncated:
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            episode_invalids.append(ep_inv)
            episode_success.append(1 if info.get("both_arrived", False) else 0)
            episode_collisions.append(1 if info.get("collision", False) else 0)
            episode_timeouts.append(1 if (truncated and not info.get("both_arrived", False) and not info.get("collision", False)) else 0)

            # Reset for next episode
            pos_a, pos_b = env.reset()
            state_feat = joint_state_to_features(pos_a, pos_b)
            ep_ret = 0.0
            ep_len = 0
            ep_inv = 0

    history = {
        "episode_returns": np.array(episode_returns),
        "episode_lengths": np.array(episode_lengths),
        "episode_success": np.array(episode_success),
        "episode_collisions": np.array(episode_collisions),
        "episode_timeouts": np.array(episode_timeouts),
        "episode_invalids": np.array(episode_invalids),
        "losses": np.array(loss_history),
        "total_optimizer_updates": agent.optimizer_updates
    }
    return agent, history


def evaluate_dqn_two_robots_greedy(
    agent: TwoRobotDQNAgent,
    start_a: Tuple[int, int] = (2, 0),
    start_b: Tuple[int, int] = (2, 4),
    max_steps: int = 40
) -> Dict[str, Any]:
    """Greedy rollout for Two-Robot Controller with epsilon=0."""
    env = TwoRobotWarehouseEnv(start_a=start_a, start_b=start_b, max_steps=max_steps)
    pos_a, pos_b = env.reset(start_a=start_a, start_b=start_b)
    state_feat = joint_state_to_features(pos_a, pos_b)

    step_log = []
    path_a = [pos_a]
    path_b = [pos_b]
    total_reward = 0.0
    total_invalids = 0
    done = False

    while not done:
        joint_act = agent.select_action(state_feat, epsilon=0.0, training=False)
        act_a, act_b = divmod(joint_act, 5)

        (next_pos_a, next_pos_b), reward, terminated, truncated, info = env.step(joint_act)
        total_reward += reward
        total_invalids += info["invalid_moves"]

        event_str = "Normal move"
        if info.get("collision", False):
            event_str = f"COLLISION ({info['collision_type']})"
        elif info.get("both_arrived", False):
            event_str = "BOTH ARRIVED (Goal)"
        elif info.get("newly_arrived", 0) > 0:
            if info["a_arrived"] and not info["b_arrived"]:
                event_str = "Robot A arrived at goal"
            elif info["b_arrived"] and not info["a_arrived"]:
                event_str = "Robot B arrived at goal"
            else:
                event_str = "Robot arrived at goal"

        step_log.append({
            "step": env.step_count,
            "pos_a": pos_a,
            "action_a": ACTION_NAMES[act_a],
            "pos_b": pos_b,
            "action_b": ACTION_NAMES[act_b],
            "next_pos_a": next_pos_a,
            "next_pos_b": next_pos_b,
            "reward": reward,
            "event": event_str
        })

        path_a.append(next_pos_a)
        path_b.append(next_pos_b)

        pos_a, pos_b = next_pos_a, next_pos_b
        state_feat = joint_state_to_features(pos_a, pos_b)
        done = terminated or truncated

    outcome = "timeout"
    if info.get("both_arrived", False):
        outcome = "success"
    elif info.get("collision", False):
        outcome = "collision"

    return {
        "outcome": outcome,
        "steps": env.step_count,
        "total_reward": total_reward,
        "invalid_moves": total_invalids,
        "path_a": path_a,
        "path_b": path_b,
        "step_log": step_log,
        "a_arrived": info.get("a_arrived", False),
        "b_arrived": info.get("b_arrived", False),
        "both_arrived": info.get("both_arrived", False),
        "collision": info.get("collision", False)
    }


def plot_two_robot_results(history: Dict[str, Any], save_path: str = "two_robot_training.png"):
    """Plot moving average curves for return, success, collision, and timeout."""
    returns = history["episode_returns"]
    successes = history["episode_success"]
    collisions = history["episode_collisions"]
    timeouts = history["episode_timeouts"]

    window = min(50, max(5, len(returns) // 10))

    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode='valid')

    ret_ma = moving_avg(returns, window)
    succ_ma = moving_avg(successes, window)
    coll_ma = moving_avg(collisions, window)
    time_ma = moving_avg(timeouts, window)
    x_vals = np.arange(window, len(returns) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(returns, alpha=0.2, color="gray", label="Raw Episode Return")
    ax1.plot(x_vals, ret_ma, color="navy", lw=2, label=f"Return ({window}-ep MA)")
    ax1.set_ylabel("Return")
    ax1.set_title("Two-Robot DQN Training Progress (60,000 Steps, Seed 7)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(x_vals, succ_ma * 100.0, color="forestgreen", lw=2, label=f"Success Rate ({window}-ep MA)")
    ax2.plot(x_vals, coll_ma * 100.0, color="crimson", lw=2, label=f"Collision Rate ({window}-ep MA)")
    ax2.plot(x_vals, time_ma * 100.0, color="darkorange", lw=2, label=f"Timeout Rate ({window}-ep MA)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved two-robot training plot to {save_path}")


def plot_trajectories(eval_res: Dict[str, Any], save_path: str = "two_robot_trajectory.png"):
    """Plot the spatial paths of Robot A and Robot B with timestep markers."""
    path_a = eval_res["path_a"]
    path_b = eval_res["path_b"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Grid lines
    for r in range(6):
        ax.axhline(r - 0.5, color="lightgray", lw=1)
    for c in range(6):
        ax.axvline(c - 0.5, color="lightgray", lw=1)

    # Obstacles
    for r, c in OBSTACLES:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="dimgray", alpha=0.85))
        ax.text(c, r, "Obstacle", ha="center", va="center", color="white", fontsize=8, weight="bold")

    # Start and Goal markers
    ax.scatter(0, 2, s=200, marker="s", color="royalblue", zorder=4, label="Start A (2,0)")
    ax.scatter(4, 2, s=200, marker="*", color="royalblue", zorder=4, label="Goal A (2,4)")
    ax.scatter(4, 2, s=150, marker="s", color="darkorange", zorder=4, label="Start B (2,4)")
    ax.scatter(0, 2, s=150, marker="*", color="darkorange", zorder=4, label="Goal B (2,0)")

    # Plot Robot A path
    cols_a = [p[1] for p in path_a]
    rows_a = [p[0] for p in path_a]
    ax.plot(cols_a, rows_a, color="royalblue", lw=3, linestyle="-", marker="o", label="Robot A Path", zorder=5)
    for t, (r, c) in enumerate(path_a):
        ax.annotate(f"A:t={t}", (c, r), textcoords="offset points", xytext=(-15, 8), fontsize=8, color="navy", weight="bold")

    # Plot Robot B path
    cols_b = [p[1] for p in path_b]
    rows_b = [p[0] for p in path_b]
    ax.plot(cols_b, rows_b, color="darkorange", lw=3, linestyle="--", marker="^", label="Robot B Path", zorder=5)
    for t, (r, c) in enumerate(path_b):
        ax.annotate(f"B:t={t}", (c, r), textcoords="offset points", xytext=(8, -12), fontsize=8, color="saddlebrown", weight="bold")

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)  # Inverted so row 0 is top
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"Two-Robot Trajectories ({eval_res['outcome'].upper()}, {eval_res['steps']} Steps)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory visualization to {save_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    train_plot = os.path.join(out_dir, "two_robot_training.png")
    traj_plot = os.path.join(out_dir, "two_robot_trajectory.png")

    print("Training Two-Robot DQN Controller (60,000 steps, seed 7)...")
    agent, history = train_dqn_two_robots(total_steps=60000, seed=7)
    plot_two_robot_results(history, save_path=train_plot)

    print("\nGreedy Evaluation from Default Starts: A=(2,0), B=(2,4)...")
    eval_res = evaluate_dqn_two_robots_greedy(agent, start_a=(2, 0), start_b=(2, 4))
    plot_trajectories(eval_res, save_path=traj_plot)

    print("\nGreedy Evaluation Summary:")
    print(f"Outcome: {eval_res['outcome']}")
    print(f"Total Steps: {eval_res['steps']}")
    print(f"Total Reward: {eval_res['total_reward']}")
    print(f"Invalid Moves: {eval_res['invalid_moves']}")
    print(f"Both Arrived: {eval_res['both_arrived']}")
    print(f"Collision: {eval_res['collision']}")

    print("\nStep-by-Step Joint Trajectory Table:")
    print(f"{'Step':<5} | {'Pos A':<8} | {'Act A':<7} | {'Pos B':<8} | {'Act B':<7} | {'Reward':<7} | {'Event'}")
    print("-" * 75)
    for log in eval_res["step_log"]:
        print(f"{log['step']:<5} | {str(log['pos_a']):<8} | {log['action_a']:<7} | {str(log['pos_b']):<8} | {log['action_b']:<7} | {log['reward']:<7.1f} | {log['event']}")
