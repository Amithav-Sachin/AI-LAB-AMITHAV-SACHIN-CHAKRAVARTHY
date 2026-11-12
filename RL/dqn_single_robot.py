"""
Deep Q-Network (DQN) for Single Robot Warehouse Navigation.
Fulfills all requirements of Task 6:
- 25 one-hot inputs -> 64 ReLU -> 64 ReLU -> 5 linear outputs
- Replay buffer of capacity 20,000, batch size 64
- Collect 500 warmup steps before learning
- 1 optimizer update every 4 environment steps
- Target network updated every 250 optimizer updates
- Adam optimizer (lr=0.001), MSE TD error loss
- 12,000 environment steps, seed 7
- Epsilon decays linearly 1.0 -> 0.05 over first 80% (9,600 steps)
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
    SingleRobotWarehouseEnv, UP, DOWN, LEFT, RIGHT, WAIT,
    ACTION_NAMES, pos_to_index, index_to_pos, pos_to_one_hot,
    OBSTACLES, GRID_SIZE
)


class DQNNetwork(nn.Module):
    """Feedforward Q-Network: 25 -> 64 -> 64 -> 5."""

    def __init__(self, in_features: int = 25, hidden_dim: int = 64, num_actions: int = 5):
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


class ReplayBuffer:
    """Cyclic replay buffer for storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_one_hot: np.ndarray, action: int, reward: float, next_state_one_hot: np.ndarray, done: bool):
        self.buffer.append((state_one_hot, action, reward, next_state_one_hot, done))

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


class DQNAgent:
    """DQN Agent managing online and target networks, action selection, and updates."""

    def __init__(
        self,
        in_features: int = 25,
        num_actions: int = 5,
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

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.online_net = DQNNetwork(in_features, 64, num_actions)
        self.target_net = DQNNetwork(in_features, 64, num_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        self.optimizer_updates = 0

    def select_action(self, state_one_hot: np.ndarray, epsilon: float, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            s_t = torch.tensor(state_one_hot, dtype=torch.float32).unsqueeze(0)
            q_vals = self.online_net(s_t).squeeze(0).numpy()
            return int(np.argmax(q_vals))

    def update_step(self) -> Optional[float]:
        """Perform one gradient step using a random minibatch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 1. Online Q-values for recorded actions
        # Where the recorded action is selected: gather along action dimension
        q_all = self.online_net(states)
        q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Target Q-values: detached with torch.no_grad()
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            max_next_q, _ = next_q_all.max(dim=1)
            # Terminal outcomes suppress continuation; truncation does NOT
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.optimizer_updates += 1
        # Synchronize target network every 250 optimizer updates
        if self.optimizer_updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()


def get_dqn_epsilon(step: int, total_steps: int = 12000, decay_fraction: float = 0.80, eps_start: float = 1.0, eps_end: float = 0.05) -> float:
    """Linear epsilon decay over the first 80% of environment steps."""
    decay_steps = int(total_steps * decay_fraction)
    if step >= decay_steps:
        return eps_end
    return eps_start - (eps_start - eps_end) * (step / decay_steps)


def train_dqn_single_robot(
    total_steps: int = 12000,
    warmup_steps: int = 500,
    update_interval: int = 4,
    target_update_freq: int = 250,
    lr: float = 0.001,
    seed: int = 7
) -> Tuple[DQNAgent, Dict[str, Any]]:
    """Train single-robot DQN agent for 12,000 environment steps."""
    env = SingleRobotWarehouseEnv()
    agent = DQNAgent(
        in_features=25,
        num_actions=5,
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
    episode_invalids: List[int] = []
    loss_history: List[float] = []
    step_epsilons: List[float] = []

    state = env.reset()
    state_oh = pos_to_one_hot(state)
    ep_ret = 0.0
    ep_len = 0
    ep_inv = 0

    for step in range(1, total_steps + 1):
        eps = get_dqn_epsilon(step, total_steps=total_steps, decay_fraction=0.80)
        step_epsilons.append(eps)

        action = agent.select_action(state_oh, eps, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_oh = pos_to_one_hot(next_state)

        ep_ret += reward
        ep_len += 1
        if info["invalid_move"]:
            ep_inv += 1

        # Store transition: 'dones' is True ONLY for true termination (goal reached),
        # allowing continuation bootstrapping at truncation.
        agent.memory.push(state_oh, action, reward, next_state_oh, terminated)

        state = next_state
        state_oh = next_state_oh

        # Train every 4 steps once warmup is complete
        if step >= warmup_steps and step % update_interval == 0:
            loss = agent.update_step()
            if loss is not None:
                loss_history.append(loss)

        if terminated or truncated:
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            episode_success.append(1 if terminated else 0)
            episode_invalids.append(ep_inv)

            # Reset environment for next episode
            state = env.reset()
            state_oh = pos_to_one_hot(state)
            ep_ret = 0.0
            ep_len = 0
            ep_inv = 0

    history = {
        "episode_returns": np.array(episode_returns),
        "episode_lengths": np.array(episode_lengths),
        "episode_success": np.array(episode_success),
        "episode_invalids": np.array(episode_invalids),
        "losses": np.array(loss_history),
        "total_optimizer_updates": agent.optimizer_updates
    }
    return agent, history


def evaluate_dqn_greedy(
    agent: DQNAgent,
    start_pos: Tuple[int, int] = (2, 0),
    max_steps: int = 40
) -> Dict[str, Any]:
    """Greedy rollout with exploration disabled (epsilon = 0.0)."""
    env = SingleRobotWarehouseEnv(start=start_pos, max_steps=max_steps)
    state = env.reset(start_pos=start_pos)
    state_oh = pos_to_one_hot(state)

    path = [state]
    actions = []
    rewards = []
    invalids = 0
    done = False

    while not done:
        action = agent.select_action(state_oh, epsilon=0.0, training=False)
        actions.append(action)
        next_state, r, terminated, truncated, info = env.step(action)
        path.append(next_state)
        rewards.append(r)
        if info["invalid_move"]:
            invalids += 1
        state = next_state
        state_oh = pos_to_one_hot(state)
        done = terminated or truncated

    return {
        "start": start_pos,
        "path": path,
        "actions": actions,
        "total_return": sum(rewards),
        "steps": env.step_count,
        "reached_goal": path[-1] == env.goal,
        "invalid_moves": invalids,
        "timeout": (env.step_count >= max_steps) and (path[-1] != env.goal)
    }


def plot_dqn_learning_curves(history: Dict[str, Any], save_path: str = "dqn_training.png"):
    """Plot DQN training return and success rate moving averages."""
    returns = history["episode_returns"]
    successes = history["episode_success"]
    window = min(30, max(5, len(returns) // 10))

    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode='valid')

    ret_ma = moving_avg(returns, window)
    succ_ma = moving_avg(successes, window)
    x_vals = np.arange(window, len(returns) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.plot(returns, alpha=0.3, color="mediumseagreen", label="Episode Return")
    ax1.plot(x_vals, ret_ma, color="darkgreen", lw=2, label=f"{window}-Episode MA Return")
    ax1.set_ylabel("Return")
    ax1.set_title("Single-Robot DQN Training Progress (12,000 Steps, Seed 7)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(x_vals, succ_ma * 100.0, color="dodgerblue", lw=2, label=f"Success Rate ({window}-ep MA)")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    if len(history["losses"]) > 0:
        loss_ma = moving_avg(history["losses"], min(100, len(history["losses"])))
        ax3.plot(loss_ma, color="crimson", lw=1.5, label="TD Loss (MA)")
    ax3.set_ylabel("TD Loss")
    ax3.set_xlabel("Update Step")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved DQN training plot to {save_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(out_dir, "dqn_training.png")

    print("Training Single-Robot DQN (12,000 steps, seed 7)...")
    agent, history = train_dqn_single_robot(total_steps=12000, seed=7)
    plot_dqn_learning_curves(history, save_path=plot_file)

    eval_res = evaluate_dqn_greedy(agent, start_pos=(2, 0))
    print("\nDQN Greedy Evaluation from Start (2, 0):")
    print(f"Reached goal: {eval_res['reached_goal']}")
    print(f"Steps: {eval_res['steps']}")
    print(f"Total return: {eval_res['total_return']}")
    print(f"Invalid moves: {eval_res['invalid_moves']}")
    print(f"Trajectory: {eval_res['path']}")
    print("Actions:", [ACTION_NAMES[a] for a in eval_res['actions']])
