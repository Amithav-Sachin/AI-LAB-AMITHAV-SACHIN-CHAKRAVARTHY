"""
Test suite for Warehouse Environment and Synthetic Q-learning / Loss Updates.
Implements all unit tests required by Task 4 and Task 7.
"""

import numpy as np
from warehouse_env import (
    SingleRobotWarehouseEnv, TwoRobotWarehouseEnv,
    UP, DOWN, LEFT, RIGHT, WAIT,
    pos_to_index, index_to_pos, pos_to_one_hot
)


def test_task4_single_robot_env():
    print("=" * 70)
    print("TASK 4: Testing Single Robot Warehouse Environment")
    print("=" * 70)
    env = SingleRobotWarehouseEnv()

    test_cases = [
        # (initial_pos, action, expected_pos, expected_r, expected_term, desc)
        ((0, 0), UP, (0, 0), -1.0, False, "Boundary move from (0,0) UP"),
        ((1, 1), RIGHT, (1, 1), -1.0, False, "Obstacle move into (1,2) from (1,1) RIGHT"),
        ((2, 1), RIGHT, (2, 2), -0.1, False, "Valid move into doorway (2,2) from (2,1) RIGHT"),
        ((2, 1), WAIT, (2, 1), -0.1, False, "Valid wait at (2,1)"),
        ((2, 3), RIGHT, (2, 4), 10.0, True, "Reach goal at (2,4) from (2,3) RIGHT"),
    ]

    all_passed = True
    for init_pos, action, exp_pos, exp_r, exp_term, desc in test_cases:
        env.reset(start_pos=init_pos)
        next_pos, r, term, trunc, info = env.step(action)
        passed = (next_pos == exp_pos) and np.isclose(r, exp_r) and (term == exp_term)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {desc}:")
        print(f"       Init={init_pos}, Action={action} -> Pos={next_pos} (exp {exp_pos}), Reward={r} (exp {exp_r}), Term={term} (exp {exp_term})")

    # Check 40-step rollout truncation
    env.reset(start_pos=(2, 0))
    truncated = False
    for step in range(40):
        # execute wait
        _, _, term, trunc, _ = env.step(WAIT)
        if trunc:
            truncated = True
            break
    print(f"[{'PASS' if truncated and step == 39 else 'FAIL'}] 40-step rollout truncation: stopped at step {step+1}, truncated={truncated}")
    if not (truncated and step == 39):
        all_passed = False

    return all_passed


def test_task4_synthetic_q_updates():
    print("\n" + "=" * 70)
    print("TASK 4: Testing Synthetic Q-learning Updates")
    print("=" * 70)
    alpha = 0.2
    gamma = 0.95
    q_sa = 0.4

    # Case 1: Non-terminal (r=1, max Q(s',a')=2)
    # Hand target: y = 1 + 0.95 * 2 = 2.9
    # Hand updated: Q(s,a) = 0.4 + 0.2 * (2.9 - 0.4) = 0.4 + 0.5 = 0.9
    r1 = 1.0
    max_next_q1 = 2.0
    y1 = r1 + gamma * max_next_q1
    q1_updated = q_sa + alpha * (y1 - q_sa)
    print(f"Non-terminal Case: r={r1}, max_q'={max_next_q1}")
    print(f"  Target: y = {r1} + {gamma} * {max_next_q1} = {y1:.4f}")
    print(f"  Updated Q: {q_sa} + {alpha} * ({y1:.4f} - {q_sa}) = {q1_updated:.4f}")
    assert np.isclose(y1, 2.9) and np.isclose(q1_updated, 0.9), "Synthetic non-terminal update mismatch!"

    # Case 2: Truly terminal with r=5
    # Hand target: y = 5
    # Hand updated: Q(s,a) = 0.4 + 0.2 * (5.0 - 0.4) = 0.4 + 0.92 = 1.32
    r2 = 5.0
    y2 = r2  # terminal suppresses max Q(s',a')
    q2_updated = q_sa + alpha * (y2 - q_sa)
    print(f"Terminal Case: r={r2} (ignores max_q')")
    print(f"  Target: y = {r2:.4f}")
    print(f"  Updated Q: {q_sa} + {alpha} * ({r2:.4f} - {q_sa}) = {q2_updated:.4f}")
    assert np.isclose(y2, 5.0) and np.isclose(q2_updated, 1.32), "Synthetic terminal update mismatch!"

    # Case 3: Truncation flag set with r=1, max Q(s',a')=2
    # Hand target: y = 1 + 0.95 * 2 = 2.9 (bootstraps because cutoff is not task termination)
    # Hand updated: Q(s,a) = 0.9
    r3 = 1.0
    y3 = r3 + gamma * max_next_q1
    q3_updated = q_sa + alpha * (y3 - q_sa)
    print(f"Truncated Case: r={r3}, max_q'={max_next_q1} (bootstraps at cutoff)")
    print(f"  Target: y = {r3} + {gamma} * {max_next_q1} = {y3:.4f}")
    print(f"  Updated Q: {q_sa} + {alpha} * ({y3:.4f} - {q_sa}) = {q3_updated:.4f}")
    assert np.isclose(y3, 2.9) and np.isclose(q3_updated, 0.9), "Synthetic truncated update mismatch!"

    print("Question answer: The TRULY TERMINAL case ignores the next-state value.")
    return True


def test_task7_two_robot_env():
    print("\n" + "=" * 70)
    print("TASK 7: Testing Two-Robot Collision and Movement Dynamics")
    print("=" * 70)
    env = TwoRobotWarehouseEnv()

    # Table from PDF page 10:
    # 1. Old A,B: (2,1),(2,3) | A: right, B: left -> Same-cell collision at (2,2)
    env.set_state((2, 1), (2, 3))
    joint_act1 = 5 * RIGHT + LEFT
    next_state, r, term, trunc, info = env.step(joint_act1)
    pass1 = (term is True) and (r == -10.0) and (next_state == ((2, 1), (2, 3))) and (info["collision_type"] == "same_cell")
    print(f"[{'PASS' if pass1 else 'FAIL'}] Test 1 ((2,1),(2,3) right,left): Collided={info['collision']}, type={info['collision_type']}, r={r}, pos={next_state}")

    # 2. Old A,B: (2,1),(2,2) | A: right, B: left -> Swap collision
    env.set_state((2, 1), (2, 2))
    joint_act2 = 5 * RIGHT + LEFT
    next_state, r, term, trunc, info = env.step(joint_act2)
    pass2 = (term is True) and (r == -10.0) and (next_state == ((2, 1), (2, 2))) and (info["collision_type"] == "swap")
    print(f"[{'PASS' if pass2 else 'FAIL'}] Test 2 ((2,1),(2,2) right,left): Collided={info['collision']}, type={info['collision_type']}, r={r}, pos={next_state}")

    # 3. Old A,B: (2,1),(2,2) | A: right, B: right -> Follow move (allowed, no collision)
    env.set_state((2, 1), (2, 2))
    joint_act3 = 5 * RIGHT + RIGHT
    next_state, r, term, trunc, info = env.step(joint_act3)
    pass3 = (term is False) and (not info["collision"]) and (next_state == ((2, 2), (2, 3))) and np.isclose(r, -0.1)
    print(f"[{'PASS' if pass3 else 'FAIL'}] Test 3 ((2,1),(2,2) right,right): Collided={info['collision']}, r={r}, pos={next_state}")

    # 4. Old A,B: (2,1),(2,2) | A: up, B: wait -> A moves to (1,1), B stays at (2,2) (allowed)
    env.set_state((2, 1), (2, 2))
    joint_act4 = 5 * UP + WAIT
    next_state, r, term, trunc, info = env.step(joint_act4)
    pass4 = (term is False) and (not info["collision"]) and (next_state == ((1, 1), (2, 2))) and np.isclose(r, -0.1)
    print(f"[{'PASS' if pass4 else 'FAIL'}] Test 4 ((2,1),(2,2) up,wait): Collided={info['collision']}, r={r}, pos={next_state}")

    # 5. Robot already at goal: stays there and cannot receive repeated arrival rewards
    env.set_state((2, 4), (2, 2))  # A is at goal (2,4)
    env.a_arrived = True
    env.b_arrived = False
    joint_act5 = 5 * LEFT + WAIT   # A attempts left, B waits
    next_state, r, term, trunc, info = env.step(joint_act5)
    pass5 = (next_state[0] == (2, 4)) and (info["newly_arrived"] == 0) and np.isclose(r, -0.1)
    print(f"[{'PASS' if pass5 else 'FAIL'}] Test 5 (Robot A already at goal): posA={next_state[0]} (stays at (2,4)), newly_arrived={info['newly_arrived']}, r={r}")

    all_pass = pass1 and pass2 and pass3 and pass4 and pass5
    print(f"\nTwo-robot test suite overall: {'ALL PASSED' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


if __name__ == "__main__":
    t4_env = test_task4_single_robot_env()
    t4_q = test_task4_synthetic_q_updates()
    t7_env = test_task7_two_robot_env()
    print("\n" + "=" * 70)
    print(f"Summary: Task 4 Env: {t4_env}, Task 4 Q-updates: {t4_q}, Task 7 Env: {t7_env}")
    print("=" * 70)
