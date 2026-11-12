"""
Unit Test for DQN: Synthetic Batch Loss, TD Targets, and Gradient Isolation (Task 6).
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class DummyQNetwork(nn.Module):
    """Simple linear mock network for exact mathematical verification."""
    def __init__(self, in_features=25, out_features=5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        # Initialize weights to fixed constant so Q(s, a) = 0.4
        # If input is one-hot (sum=1), setting weight = 0.4 gives output = 0.4
        nn.init.constant_(self.fc.weight, 0.4)

    def forward(self, x):
        return self.fc(x)


class DummyTargetNetwork(nn.Module):
    """Simple mock target network that returns constant 2.0 for all Q-values."""
    def __init__(self, in_features=25, out_features=5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        nn.init.constant_(self.fc.weight, 2.0)

    def forward(self, x):
        return self.fc(x)


def compute_dqn_loss(
    online_net: nn.Module,
    target_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,  # 1.0 for true termination, 0.0 otherwise
    gamma: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes DQN MSE loss.
    Uses torch.no_grad() for target network computation.
    Gathers recorded actions from online network.
    """
    # 1. Online Q-values for recorded actions
    # online_net(states) has shape (B, num_actions)
    # gather selects the element at actions index: (B, 1) -> squeezed to (B,)
    q_all = online_net(states)
    q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. Target Q-values
    with torch.no_grad():
        target_next_q_all = target_net(next_states)
        max_next_q, _ = target_next_q_all.max(dim=1)
        # True termination suppresses continuation: (1 - dones) * max_next_q
        targets = rewards + gamma * (1.0 - dones) * max_next_q

    # 3. MSE Loss
    loss = nn.MSELoss()(q_selected, targets)
    return loss, targets, q_selected


def test_synthetic_dqn_batch():
    print("=" * 70)
    print("TASK 6: Testing Synthetic DQN Batch TD Targets and Gradient Isolation")
    print("=" * 70)

    gamma = 0.95
    online_net = DummyQNetwork()
    target_net = DummyTargetNetwork()

    # Batch of 3 items:
    # Item 0: Non-terminal: r = 1, next max Q = 2, done = 0
    # Item 1: Terminal:     r = 5, done = 1
    # Item 2: Truncated:    r = 1, next max Q = 2, done = 0 (truncation bootstraps, done = 0)
    states = torch.zeros(3, 25)
    states[0, 0] = 1.0
    states[1, 5] = 1.0
    states[2, 10] = 1.0

    actions = torch.tensor([1, 2, 3], dtype=torch.long)
    rewards = torch.tensor([1.0, 5.0, 1.0], dtype=torch.float32)

    next_states = torch.zeros(3, 25)
    next_states[0, 1] = 1.0
    next_states[1, 6] = 1.0
    next_states[2, 11] = 1.0

    dones = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    loss, targets, q_selected = compute_dqn_loss(
        online_net, target_net, states, actions, rewards, next_states, dones, gamma=gamma
    )

    targets_np = targets.numpy()
    q_np = q_selected.detach().numpy()
    loss_val = loss.item()

    # Hand calculation:
    # Target 0: 1.0 + 0.95 * 2.0 = 2.90
    # Target 1: 5.0 (terminal suppresses continuation)
    # Target 2: 1.0 + 0.95 * 2.0 = 2.90 (truncation bootstraps continuation)
    exp_targets = [2.90, 5.00, 2.90]
    # Online Q are all 0.40
    # errors = [-2.50, -4.60, -2.50]
    # squared = [6.25, 21.16, 6.25]
    # mean = (6.25 + 21.16 + 6.25) / 3 = 33.66 / 3 = 11.22
    exp_loss = 11.22

    print(f"Computed Targets: {targets_np} | Expected: {exp_targets}")
    print(f"Selected Q(s, a): {q_np} | Expected: [0.4, 0.4, 0.4]")
    print(f"Computed Loss:    {loss_val:.6f} | Expected: {exp_loss:.6f}")

    targets_match = np.allclose(targets_np, exp_targets)
    loss_match = np.isclose(loss_val, exp_loss)
    print(f"[{'PASS' if targets_match else 'FAIL'}] TD Target calculation")
    print(f"[{'PASS' if loss_match else 'FAIL'}] MSE Loss calculation")

    # Test Gradient isolation:
    loss.backward()
    online_has_grad = all(p.grad is not None for p in online_net.parameters())
    target_has_no_grad = all(p.grad is None for p in target_net.parameters())

    print(f"[{'PASS' if online_has_grad else 'FAIL'}] Online network received gradients")
    print(f"[{'PASS' if target_has_no_grad else 'FAIL'}] Target network isolated (no gradients flowed into target)")

    overall = targets_match and loss_match and online_has_grad and target_has_no_grad
    print(f"\nOverall Synthetic DQN Batch Test: {'ALL PASSED' if overall else 'FAILED'}")
    return overall


if __name__ == "__main__":
    test_synthetic_dqn_batch()
