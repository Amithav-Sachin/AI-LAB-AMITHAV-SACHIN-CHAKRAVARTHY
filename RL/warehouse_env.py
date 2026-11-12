"""
Warehouse Environment for CS F407 Reinforcement Learning Laboratory.
Implements:
1. Single-Robot Warehouse MDP (5x5 grid with doorway obstacle wall).
2. Two-Robot Warehouse Joint Controller MDP with collision detection.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np

# Action constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WAIT = 4

ACTION_NAMES = {
    UP: "Up",
    DOWN: "Down",
    LEFT: "Left",
    RIGHT: "Right",
    WAIT: "Wait"
}

ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    WAIT: (0, 0)
}

OBSTACLES = {(0, 2), (1, 2), (3, 2), (4, 2)}
GRID_SIZE = 5
MAX_STEPS = 40


def pos_to_index(pos: Tuple[int, int]) -> int:
    """Convert (row, col) to linear index 5*row + col."""
    return 5 * pos[0] + pos[1]


def index_to_pos(idx: int) -> Tuple[int, int]:
    """Convert linear index to (row, col)."""
    return divmod(idx, 5)


def pos_to_one_hot(pos: Tuple[int, int], size: int = 25) -> np.ndarray:
    """Convert (row, col) to length-25 one-hot vector."""
    vec = np.zeros(size, dtype=np.float32)
    vec[pos_to_index(pos)] = 1.0
    return vec


class SingleRobotWarehouseEnv:
    """
    Single-robot warehouse environment on a 5x5 grid.
    Start: (2, 0)
    Goal: (2, 4)
    Obstacles: {(0, 2), (1, 2), (3, 2), (4, 2)}
    Doorway at (2, 2)
    """

    def __init__(self, start: Tuple[int, int] = (2, 0), goal: Tuple[int, int] = (2, 4), max_steps: int = MAX_STEPS):
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.obstacles = set(OBSTACLES)
        self.grid_size = GRID_SIZE
        self.robot_pos = self.start
        self.step_count = 0

    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Reset environment to start position."""
        self.robot_pos = self.start if start_pos is None else start_pos
        self.step_count = 0
        return self.robot_pos

    def set_state(self, pos: Tuple[int, int]):
        """Directly set the robot position for testing without resetting step count."""
        self.robot_pos = pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.
        Returns:
            next_state: Tuple[int, int]
            reward: float
            terminated: bool (true termination at goal)
            truncated: bool (episode cutoff after 40 steps)
            info: dict (details: invalid_move, is_goal, etc.)
        """
        assert action in (UP, DOWN, LEFT, RIGHT, WAIT), f"Invalid action: {action}"
        self.step_count += 1
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.robot_pos[0] + dr, self.robot_pos[1] + dc

        is_invalid = False
        # Check boundary collision or obstacle collision
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            is_invalid = True
        elif (nr, nc) in self.obstacles:
            is_invalid = True

        if is_invalid:
            next_pos = self.robot_pos
            reward = -1.0
            terminated = False
        else:
            next_pos = (nr, nc)
            if next_pos == self.goal:
                reward = 10.0
                terminated = True
            else:
                reward = -0.1
                terminated = False

        self.robot_pos = next_pos
        truncated = (not terminated) and (self.step_count >= self.max_steps)

        info = {
            "step": self.step_count,
            "invalid_move": is_invalid,
            "reached_goal": terminated,
            "pos": self.robot_pos
        }

        return next_pos, reward, terminated, truncated, info


class TwoRobotWarehouseEnv:
    """
    Two-robot warehouse environment with collision detection.
    Robot A: Start (2, 0), Goal (2, 4)
    Robot B: Start (2, 4), Goal (2, 0)
    Joint actions: j in [0, 24], decoded as (aA, aB) = divmod(j, 5)
    """

    def __init__(
        self,
        start_a: Tuple[int, int] = (2, 0),
        goal_a: Tuple[int, int] = (2, 4),
        start_b: Tuple[int, int] = (2, 4),
        goal_b: Tuple[int, int] = (2, 0),
        max_steps: int = MAX_STEPS
    ):
        self.start_a = start_a
        self.goal_a = goal_a
        self.start_b = start_b
        self.goal_b = goal_b
        self.max_steps = max_steps
        self.obstacles = set(OBSTACLES)
        self.grid_size = GRID_SIZE

        self.pos_a = self.start_a
        self.pos_b = self.start_b
        self.step_count = 0
        self.a_arrived = False
        self.b_arrived = False

    def reset(
        self,
        start_a: Optional[Tuple[int, int]] = None,
        start_b: Optional[Tuple[int, int]] = None
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        self.pos_a = self.start_a if start_a is None else start_a
        self.pos_b = self.start_b if start_b is None else start_b
        self.step_count = 0
        self.a_arrived = (self.pos_a == self.goal_a)
        self.b_arrived = (self.pos_b == self.goal_b)
        return self.pos_a, self.pos_b

    def set_state(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]):
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.a_arrived = (self.pos_a == self.goal_a)
        self.b_arrived = (self.pos_b == self.goal_b)

    def step(self, joint_action: int) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float, bool, bool, Dict[str, Any]]:
        """
        Execute one joint action j = 5 * aA + aB.
        Returns:
            next_state: ((rA, cA), (rB, cB))
            reward: float
            terminated: bool (collision or both reached goal)
            truncated: bool (episode cutoff after 40 steps)
            info: dict
        """
        a_a, a_b = divmod(joint_action, 5)
        self.step_count += 1

        old_a = self.pos_a
        old_b = self.pos_b

        # 1. Propose moves
        # A robot already at its goal stays, regardless of requested action.
        invalid_a = False
        if self.a_arrived:
            prop_a = old_a
        else:
            dr, dc = ACTION_DELTAS[a_a]
            nr, nc = old_a[0] + dr, old_a[1] + dc
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size) or (nr, nc) in self.obstacles:
                prop_a = old_a
                invalid_a = True
            else:
                prop_a = (nr, nc)

        invalid_b = False
        if self.b_arrived:
            prop_b = old_b
        else:
            dr, dc = ACTION_DELTAS[a_b]
            nr, nc = old_b[0] + dr, old_b[1] + dc
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size) or (nr, nc) in self.obstacles:
                prop_b = old_b
                invalid_b = True
            else:
                prop_b = (nr, nc)

        # 2. Check collision
        # Same-cell collision: proposed positions are equal
        same_cell_collision = (prop_a == prop_b)
        # Swap collision: robots exchange old positions in one simultaneous step
        swap_collision = (prop_a == old_b and prop_b == old_a and prop_a != old_a)

        is_collision = same_cell_collision or swap_collision

        if is_collision:
            # On collision, retain the old positions for reporting and terminate the episode.
            reward = -10.0
            terminated = True
            truncated = False
            # positions remain old_a, old_b
            self.pos_a = old_a
            self.pos_b = old_b
            info = {
                "step": self.step_count,
                "collision": True,
                "collision_type": "same_cell" if same_cell_collision else "swap",
                "invalid_moves": (1 if invalid_a else 0) + (1 if invalid_b else 0),
                "a_arrived": self.a_arrived,
                "b_arrived": self.b_arrived,
                "both_arrived": False
            }
            return (self.pos_a, self.pos_b), reward, terminated, truncated, info

        # If no collision: update positions
        self.pos_a = prop_a
        self.pos_b = prop_b

        # Check newly reached goals
        new_k = 0
        if (not self.a_arrived) and (self.pos_a == self.goal_a):
            self.a_arrived = True
            new_k += 1

        if (not self.b_arrived) and (self.pos_b == self.goal_b):
            self.b_arrived = True
            new_k += 1

        both_arrived = self.a_arrived and self.b_arrived
        I_val = 1.0 if both_arrived else 0.0

        b_val = (1 if invalid_a else 0) + (1 if invalid_b else 0)

        # Reward: -0.1 - b + 5k + 5I
        reward = -0.1 - float(b_val) + 5.0 * float(new_k) + 5.0 * I_val

        # Terminate if both robots have arrived
        terminated = both_arrived
        truncated = (not terminated) and (self.step_count >= self.max_steps)

        info = {
            "step": self.step_count,
            "collision": False,
            "collision_type": None,
            "invalid_moves": b_val,
            "newly_arrived": new_k,
            "a_arrived": self.a_arrived,
            "b_arrived": self.b_arrived,
            "both_arrived": both_arrived
        }

        return (self.pos_a, self.pos_b), reward, terminated, truncated, info
