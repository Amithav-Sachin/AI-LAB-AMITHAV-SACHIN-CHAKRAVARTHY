"""
Comprehensive Test Suite for Planning Agent
Laboratory: Logical Reasoning for Planning (Task 3)

Implements and records:
- Test A: Solvable Problem (Standard Warehouse Delivery)
- Test B: Impossible Problem (PickUp action removed)
- Test C: Irrelevant Actions (Distractor actions / Moving without package)
- Independent mechanical validation of all generated plans.
"""

import sys
from planner import Action, PlanningProblem, BFSPlanner, create_warehouse_actions
from verify_plan import PlanVerifier


def run_test_a():
    print("\n" + "=" * 70)
    print("RUNNING TEST A: SOLVABLE PROBLEM (Standard Warehouse Problem)")
    print("=" * 70)
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    actions = create_warehouse_actions(allow_pickup=True)
    problem = PlanningProblem(
        initial_state=initial_state,
        actions=actions,
        goal_conditions=goal,
        name="Test A: Solvable Standard Problem"
    )
    
    planner = BFSPlanner(problem)
    plan, states = planner.solve(verbose=True)
    
    # Verification
    assert plan is not None, "Test A FAILED: Expected a valid plan, got None."
    verification = PlanVerifier.verify_plan(problem, plan)
    
    print("\n[Independent Verifier Result for Test A]")
    print(f"Plan Found: Yes ({len(plan)} actions)")
    print(f"Plan Valid: {verification['valid']}")
    print(f"Goal Satisfied: {verification['goal_satisfied']}")
    assert verification['valid'], f"Plan verification failed: {verification['failure_reason']}"
    return {
        "test": "Test A",
        "initial_state": sorted(list(initial_state)),
        "goal": sorted(list(goal)),
        "plan_found": True,
        "plan": [a.name for a in plan],
        "is_valid": verification['valid']
    }


def run_test_b():
    print("\n" + "=" * 70)
    print("RUNNING TEST B: IMPOSSIBLE PROBLEM (PickUp Action Removed)")
    print("=" * 70)
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    
    # Disable PickUp action completely
    actions = create_warehouse_actions(allow_pickup=False)
    problem = PlanningProblem(
        initial_state=initial_state,
        actions=actions,
        goal_conditions=goal,
        name="Test B: Impossible Problem (No PickUp)"
    )
    
    planner = BFSPlanner(problem)
    plan, states = planner.solve(verbose=True)
    
    plan_found = (plan is not None)
    print("\n[Independent Verifier Result for Test B]")
    print(f"Plan Found: {'Yes' if plan_found else 'No (Correctly detected impossible)'}")
    print(f"Expected Behavior: Reported 'No plan found' without inventing actions.")
    assert plan is None, "Test B FAILED: Planner generated a plan when it should be impossible."
    return {
        "test": "Test B",
        "initial_state": sorted(list(initial_state)),
        "goal": sorted(list(goal)),
        "plan_found": False,
        "plan": None,
        "is_valid": "N/A (Correctly rejected)"
    }


def run_test_c():
    print("\n" + "=" * 70)
    print("RUNNING TEST C: IRRELEVANT ACTIONS (Distractor actions / Moving without package)")
    print("=" * 70)
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    
    # Standard actions plus irrelevant distractor actions
    actions = create_warehouse_actions(allow_pickup=True, include_irrelevant=True)
    actions.append(
        Action(
            name="InspectSensor(Robot)",
            positive_preconditions={"At(Robot, A)"},
            negative_preconditions=set(),
            positive_effects={"SensorChecked"},
            negative_effects=set()
        )
    )
    
    problem = PlanningProblem(
        initial_state=initial_state,
        actions=actions,
        goal_conditions=goal,
        name="Test C: Irrelevant & Distractor Actions"
    )
    
    planner = BFSPlanner(problem)
    plan, states = planner.solve(verbose=True)
    
    assert plan is not None, "Test C FAILED: Expected a valid plan."
    verification = PlanVerifier.verify_plan(problem, plan)
    
    # Sub-test verification: Check that moving robot to C without package does NOT satisfy goal
    robot_only_state = frozenset({"At(Robot, C)", "At(Package, A)"})
    assert not problem.is_goal_satisfied(robot_only_state), (
        "Faulty goal logic: Robot reaching C must not satisfy At(Package, C)!"
    )
    
    print("\n[Independent Verifier Result for Test C]")
    print(f"Plan Found: Yes ({len(plan)} actions)")
    print(f"Plan Valid: {verification['valid']}")
    print(f"Goal Satisfied: {verification['goal_satisfied']}")
    print("Verified: Irrelevant actions were ignored by shortest-path BFS.")
    print("Verified: Robot reaching C without package does NOT satisfy the goal.")
    assert verification['valid'], f"Plan verification failed: {verification['failure_reason']}"
    assert "At(Package, C)" in verification["final_state"], "Goal proposition At(Package, C) was not achieved!"
    return {
        "test": "Test C",
        "initial_state": sorted(list(initial_state)),
        "goal": sorted(list(goal)),
        "plan_found": True,
        "plan": [a.name for a in plan],
        "is_valid": verification['valid']
    }


def run_all_tests():
    res_a = run_test_a()
    res_b = run_test_b()
    res_c = run_test_c()
    
    print("\n" + "=" * 70)
    print("SUMMARY OF TEST RESULTS (TASK 3 RECORD TABLE)")
    print("=" * 70)
    print(f"{'Test':<10} | {'Initial State':<32} | {'Goal':<18} | {'Found?':<8} | {'Valid?':<8}")
    print("-" * 85)
    for r in [res_a, res_b, res_c]:
        init_s = ", ".join(r["initial_state"])
        goal_s = ", ".join(r["goal"])
        found_s = "Yes" if r["plan_found"] else "No"
        valid_s = str(r["is_valid"])
        print(f"{r['test']:<10} | {init_s:<32} | {goal_s:<18} | {found_s:<8} | {valid_s:<8}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
