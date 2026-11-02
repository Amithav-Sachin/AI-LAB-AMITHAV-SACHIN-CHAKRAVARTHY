"""
Logical Planning Agent in Python
Implements a propositional STRIPS-style planning agent using Breadth-First Search (BFS).
Laboratory: Logical Reasoning for Planning
"""

from collections import deque
from typing import Set, List, Tuple, Optional, FrozenSet


class Action:
    """
    Represents a planning action with propositional preconditions and effects.
    
    Attributes:
        name: Identifier for the action (e.g., 'Move(A, B)').
        positive_preconditions: Propositions that MUST be true (present) in the state.
        negative_preconditions: Propositions that MUST NOT be true (absent) in the state.
        positive_effects: Propositions added to the state upon execution.
        negative_effects: Propositions removed from the state upon execution.
    """
    def __init__(
        self,
        name: str,
        positive_preconditions: Optional[Set[str]] = None,
        negative_preconditions: Optional[Set[str]] = None,
        positive_effects: Optional[Set[str]] = None,
        negative_effects: Optional[Set[str]] = None,
    ):
        self.name = name
        self.positive_preconditions: FrozenSet[str] = frozenset(positive_preconditions or set())
        self.negative_preconditions: FrozenSet[str] = frozenset(negative_preconditions or set())
        self.positive_effects: FrozenSet[str] = frozenset(positive_effects or set())
        self.negative_effects: FrozenSet[str] = frozenset(negative_effects or set())

    def is_applicable(self, state: FrozenSet[str]) -> bool:
        """
        An action is applicable if all positive preconditions are present
        and no negative preconditions are present in the state:
        S |= Preconditions(a)
        """
        # All positive preconditions must be in state
        if not self.positive_preconditions.issubset(state):
            return False
        # No negative preconditions may be in state
        if bool(self.negative_preconditions.intersection(state)):
            return False
        return True

    def apply(self, state: FrozenSet[str]) -> FrozenSet[str]:
        """
        State transition function:
        1. Remove negative effects
        2. Add positive effects
        S' = (S \\ negative_effects) U positive_effects
        """
        if not self.is_applicable(state):
            raise ValueError(f"Action '{self.name}' is not applicable in state: {set(state)}")
        return (state - self.negative_effects) | self.positive_effects

    def __repr__(self) -> str:
        return f"Action({self.name})"


class PlanningProblem:
    """
    Defines a planning problem instance (I, A, G).
    """
    def __init__(
        self,
        initial_state: Set[str],
        actions: List[Action],
        goal_conditions: Set[str],
        name: str = "Planning Problem"
    ):
        self.name = name
        self.initial_state: FrozenSet[str] = frozenset(initial_state)
        self.actions: List[Action] = actions
        self.goal_conditions: FrozenSet[str] = frozenset(goal_conditions)

    def is_goal_satisfied(self, state: FrozenSet[str]) -> bool:
        """Goal is achieved if state satisfies all goal propositions."""
        return self.goal_conditions.issubset(state)


class BFSPlanner:
    """
    Breadth-First Search Planner.
    Finds the shortest sequence of actions from initial state to goal state.
    """
    def __init__(self, problem: PlanningProblem):
        self.problem = problem

    def solve(self, verbose: bool = True) -> Tuple[Optional[List[Action]], Optional[List[FrozenSet[str]]]]:
        """
        Executes BFS to find a valid plan.
        Returns:
            (plan, states_reached) if found, else (None, None).
        """
        init = self.problem.initial_state
        
        # Check if initial state already satisfies the goal
        if self.problem.is_goal_satisfied(init):
            if verbose:
                print(f"[{self.problem.name}] Initial state already satisfies goal!")
            return [], [init]

        # Queue stores: (current_state, path_of_actions, path_of_states)
        queue = deque([(init, [], [init])])
        visited: Set[FrozenSet[str]] = {init}
        nodes_explored = 0

        while queue:
            current_state, action_path, state_path = queue.popleft()
            nodes_explored += 1

            for action in self.problem.actions:
                if action.is_applicable(current_state):
                    next_state = action.apply(current_state)

                    if self.problem.is_goal_satisfied(next_state):
                        final_action_path = action_path + [action]
                        final_state_path = state_path + [next_state]
                        if verbose:
                            self._print_plan_summary(final_action_path, final_state_path, nodes_explored)
                        return final_action_path, final_state_path

                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((
                            next_state,
                            action_path + [action],
                            state_path + [next_state]
                        ))

        if verbose:
            print("=" * 60)
            print(f"PLANNING REPORT: {self.problem.name}")
            print("=" * 60)
            print(f"Result: No plan found.")
            print(f"Total states explored: {nodes_explored}")
            print(f"Total unique states visited: {len(visited)}")
            print("=" * 60)
        return None, None

    def _print_plan_summary(
        self,
        actions: List[Action],
        states: List[FrozenSet[str]],
        nodes_explored: int
    ):
        print("=" * 60)
        print(f"PLANNING REPORT: {self.problem.name}")
        print("=" * 60)
        print(f"Result: Plan found! (Length: {len(actions)} actions)")
        print(f"Search nodes explored: {nodes_explored}")
        print("-" * 60)
        print("ACTION SEQUENCE:")
        for idx, act in enumerate(actions, 1):
            print(f"  {idx}. {act.name}")
        print("-" * 60)
        print("STATE TRANSITIONS TRACE:")
        for idx, state in enumerate(states):
            state_str = ", ".join(sorted(state)) if state else "{}"
            if idx == 0:
                print(f"  S0 (Initial) : {{{state_str}}}")
            else:
                act = actions[idx - 1]
                print(f"  --[ {act.name} ]-->")
                print(f"  S{idx}          : {{{state_str}}}")
        print("=" * 60)


def create_warehouse_actions(allow_pickup: bool = True, include_irrelevant: bool = False) -> List[Action]:
    """
    Constructs the domain action schema for the warehouse world.
    Locations: A, B, C.
    Graph topology: A <-> B <-> C (A is connected to B; B is connected to C).
    """
    actions: List[Action] = []
    locations = ["A", "B", "C"]
    connected = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]

    # Move actions
    for frm, to in connected:
        actions.append(
            Action(
                name=f"Move({frm}, {to})",
                positive_preconditions={f"At(Robot, {frm})"},
                negative_preconditions=set(),
                positive_effects={f"At(Robot, {to})"},
                negative_effects={f"At(Robot, {frm})"},
            )
        )

    # PickUp actions (if enabled)
    if allow_pickup:
        for loc in locations:
            actions.append(
                Action(
                    name=f"PickUp(Package, {loc})",
                    positive_preconditions={f"At(Robot, {loc})", f"At(Package, {loc})"},
                    negative_preconditions={"Holding(Package)"},
                    positive_effects={"Holding(Package)"},
                    negative_effects={f"At(Package, {loc})"},
                )
            )

    # Drop actions
    for loc in locations:
        actions.append(
            Action(
                name=f"Drop(Package, {loc})",
                positive_preconditions={f"At(Robot, {loc})", "Holding(Package)"},
                negative_preconditions=set(),
                positive_effects={f"At(Package, {loc})"},
                negative_effects={"Holding(Package)"},
            )
        )

    # Optional irrelevant action for Test C (e.g., spin/inspect without moving package)
    if include_irrelevant:
        actions.append(
            Action(
                name="IdleBeep(Robot)",
                positive_preconditions=set(),
                negative_preconditions=set(),
                positive_effects={"RobotBeeped"},
                negative_effects=set(),
            )
        )

    return actions


def run_standard_warehouse_problem():
    """Builds and solves the standard motivating scenario."""
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    actions = create_warehouse_actions(allow_pickup=True)
    problem = PlanningProblem(
        initial_state=initial_state,
        actions=actions,
        goal_conditions=goal,
        name="Standard Warehouse Delivery (Task 2)"
    )
    planner = BFSPlanner(problem)
    return planner.solve(verbose=True)


if __name__ == "__main__":
    print("Executing Warehouse Planning Agent...")
    run_standard_warehouse_problem()
