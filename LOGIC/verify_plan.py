"""
Independent Plan Verification Engine
Laboratory: Logical Reasoning for Planning (Task 3 & Task 5)

Performs rigorous, independent, step-by-step logical validation of a plan:
1. Verifies S_k |= Preconditions(a_k) at each step k
2. Applies state update function S_{k+1} = (S_k \\ NegEff(a_k)) U PosEff(a_k)
3. Verifies S_n |= Goal
"""

from typing import List, FrozenSet, Tuple, Dict, Any
from planner import Action, PlanningProblem


class PlanVerifier:
    """
    Independent validator that mechanically checks state transitions
    and logical entailment without relying on LLM heuristics.
    """
    @staticmethod
    def verify_plan(
        problem: PlanningProblem,
        plan: List[Action]
    ) -> Dict[str, Any]:
        """
        Validates the sequence of actions against problem initial state and goal.
        Returns a detailed report.
        """
        current_state = problem.initial_state
        step_traces: List[Dict[str, Any]] = []
        is_valid = True
        failure_reason = None

        step_traces.append({
            "step": 0,
            "action": "INITIAL_STATE",
            "state_before": sorted(list(current_state)),
            "preconditions_checked": True,
            "state_after": sorted(list(current_state))
        })

        for idx, action in enumerate(plan, start=1):
            # Check positive preconditions
            missing_pos = action.positive_preconditions - current_state
            # Check negative preconditions
            violated_neg = action.negative_preconditions.intersection(current_state)

            if missing_pos or violated_neg:
                is_valid = False
                failure_reason = (
                    f"Action #{idx} '{action.name}' failed precondition check! "
                    f"Missing positive: {list(missing_pos)}, Violated negative: {list(violated_neg)}"
                )
                step_traces.append({
                    "step": idx,
                    "action": action.name,
                    "state_before": sorted(list(current_state)),
                    "preconditions_checked": False,
                    "failure_reason": failure_reason,
                    "state_after": None
                })
                break

            # Apply effects deterministically
            next_state = (current_state - action.negative_effects) | action.positive_effects
            step_traces.append({
                "step": idx,
                "action": action.name,
                "state_before": sorted(list(current_state)),
                "preconditions_checked": True,
                "state_after": sorted(list(next_state))
            })
            current_state = next_state

        goal_satisfied = False
        if is_valid:
            goal_satisfied = problem.is_goal_satisfied(current_state)
            if not goal_satisfied:
                is_valid = False
                missing_goals = problem.goal_conditions - current_state
                failure_reason = f"Plan completed execution but goal not satisfied! Missing: {list(missing_goals)}"

        return {
            "valid": is_valid,
            "goal_satisfied": goal_satisfied,
            "failure_reason": failure_reason,
            "final_state": sorted(list(current_state)),
            "steps": step_traces
        }
