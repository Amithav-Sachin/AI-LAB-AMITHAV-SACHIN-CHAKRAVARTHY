"""
Goal-Based Intelligent Agent for Warehouse Navigation
Laboratory: Agents - Constructing a Goal-Based Agent using an LLM
Course: Undergraduate Artificial Intelligence

Implements:
1. Environment model: 2D Grid with obstacles and discrete coordinates.
2. Goal-Based Agent architecture (Sensors -> State Model -> Goal Formulation -> Planning -> Actuators).
3. Search algorithms: A* (Manhattan heuristic) and BFS.
4. Path execution & step-by-step simulation.
"""

from collections import deque
import heapq
from typing import List, Tuple, Dict, Optional, Set

Coord = Tuple[int, int]

WAREHOUSE_MAP = """#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""


class WarehouseEnvironment:
    """
    Represents the physical warehouse environment.
    Provides sensory percepts to the agent and executes agent actions.
    """
    def __init__(self, map_str: str = WAREHOUSE_MAP):
        self.raw_lines = [line.strip() for line in map_str.strip().splitlines() if line.strip()]
        self.height = len(self.raw_lines)
        self.width = len(self.raw_lines[0]) if self.height > 0 else 0
        
        self.start: Optional[Coord] = None
        self.goal: Optional[Coord] = None
        self.obstacles: Set[Coord] = set()
        self._parse()

    def _parse(self):
        for r, row in enumerate(self.raw_lines):
            for c, char in enumerate(row):
                if char == 'S':
                    self.start = (r, c)
                elif char == 'G':
                    self.goal = (r, c)
                elif char == '#':
                    self.obstacles.add((r, c))

    def is_valid_coordinate(self, pos: Coord) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and (r, c) not in self.obstacles

    def get_legal_actions(self, pos: Coord) -> List[Tuple[str, Coord]]:
        """Returns legal transitions: (Action_Name, New_Coord)."""
        r, c = pos
        moves = [
            ("Up", (r - 1, c)),
            ("Down", (r + 1, c)),
            ("Left", (r, c - 1)),
            ("Right", (r, c + 1)),
        ]
        return [(act, nxt) for act, nxt in moves if self.is_valid_coordinate(nxt)]

    def render(self, agent_pos: Optional[Coord] = None, path: Optional[List[Coord]] = None) -> str:
        """Visualizes the grid with agent position and planned path."""
        grid = [list(line) for line in self.raw_lines]
        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    grid[r][c] = '*'
        if agent_pos:
            r, c = agent_pos
            grid[r][c] = 'A'
        return "\n".join("".join(row) for row in grid)


class GoalBasedAgent:
    """
    Goal-Based Intelligent Agent.
    
    Architecture (Russell & Norvig):
    - Sensors: Perceives current position and environment layout.
    - State Model: Maintains internal representation of warehouse coordinates.
    - Goal: Target destination 'G' (dispatch area).
    - Planning Engine: Synthesizes a sequence of actions that achieves the goal.
    - Actuators: Executes planned movements (Up, Down, Left, Right).
    """
    def __init__(self, env: WarehouseEnvironment, algorithm: str = "A*"):
        self.env = env
        self.algorithm = algorithm
        self.current_pos = env.start
        self.goal_pos = env.goal
        self.plan: List[Coord] = []
        self.action_history: List[str] = []
        self.nodes_expanded = 0

    def formulate_plan(self) -> bool:
        """Invokes search to find a path to the goal."""
        if self.algorithm.upper() == "A*":
            return self._plan_astar()
        else:
            return self._plan_bfs()

    def _plan_astar(self) -> bool:
        """A* Search using Manhattan distance heuristic."""
        start = self.current_pos
        goal = self.goal_pos
        if start is None or goal is None:
            return False

        def h(pos: Coord) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        counter = 0
        frontier = [(h(start), counter, start, 0)]
        g_scores: Dict[Coord, int] = {start: 0}
        parent_map: Dict[Coord, Optional[Coord]] = {start: None}
        closed_set: Set[Coord] = set()
        self.nodes_expanded = 0

        while frontier:
            f, _, curr, g = heapq.heappop(frontier)

            if curr in closed_set:
                continue
            closed_set.add(curr)
            self.nodes_expanded += 1

            if curr == goal:
                # Reconstruct path
                path = []
                step: Optional[Coord] = goal
                while step is not None:
                    path.append(step)
                    step = parent_map[step]
                path.reverse()
                self.plan = path
                return True

            for action_name, neighbor in self.env.get_legal_actions(curr):
                tentative_g = g + 1
                if neighbor in closed_set:
                    continue
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    parent_map[neighbor] = curr
                    f_score = tentative_g + h(neighbor)
                    counter += 1
                    heapq.heappush(frontier, (f_score, counter, neighbor, tentative_g))

        return False

    def _plan_bfs(self) -> bool:
        """Breadth-First Search (blind baseline)."""
        start = self.current_pos
        goal = self.goal_pos
        if start is None or goal is None:
            return False

        queue = deque([start])
        visited: Set[Coord] = {start}
        parent_map: Dict[Coord, Optional[Coord]] = {start: None}
        self.nodes_expanded = 0

        while queue:
            curr = queue.popleft()
            self.nodes_expanded += 1

            if curr == goal:
                path = []
                step: Optional[Coord] = goal
                while step is not None:
                    path.append(step)
                    step = parent_map[step]
                path.reverse()
                self.plan = path
                return True

            for action_name, neighbor in self.env.get_legal_actions(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = curr
                    queue.append(neighbor)

        return False

    def execute_plan(self, verbose: bool = False) -> List[Coord]:
        """Simulates agent executing the plan step by step."""
        execution_trace = [self.current_pos]
        for next_step in self.plan[1:]:
            self.current_pos = next_step
            execution_trace.append(self.current_pos)
            if verbose:
                print(f"Agent moved to {self.current_pos}")
        return execution_trace


def run_demo():
    env = WarehouseEnvironment()
    print("=" * 60)
    print("GOAL-BASED AGENT NAVIGATION DEMONSTRATION")
    print("=" * 60)
    print("Environment Map (21 x 7):")
    print(env.render())
    
    agent = GoalBasedAgent(env, algorithm="A*")
    success = agent.formulate_plan()
    
    if success:
        print(f"\nPlan synthesized successfully via {agent.algorithm}!")
        print(f"Path Length: {len(agent.plan) - 1} steps")
        print(f"Nodes Expanded: {agent.nodes_expanded}")
        print(f"Action Coordinates: {agent.plan}")
        print("\nVisualized Trajectory (* = planned path):")
        print(env.render(path=agent.plan))
    else:
        print("\nNo collision-free path exists to the goal!")


if __name__ == "__main__":
    run_demo()
