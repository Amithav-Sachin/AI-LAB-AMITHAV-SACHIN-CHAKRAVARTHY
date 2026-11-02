"""
Search Agent for Warehouse Robot Navigation
Laboratory: Search and A*
Implements A* Search and Breadth-First Search (BFS) over 2D grid environments.
"""

import heapq
from collections import deque
import math
from typing import List, Tuple, Dict, Optional, Set, Callable

# Coordinate representation: (row, col)
Coord = Tuple[int, int]

ORIGINAL_WAREHOUSE_MAP = """#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""


class GridEnvironment:
    """
    Represents the 2D warehouse grid parsed from an ASCII map.
    """
    def __init__(self, map_str: str, name: str = "Warehouse"):
        self.name = name
        self.raw_map = [line.strip() for line in map_str.strip().splitlines() if line.strip()]
        self.height = len(self.raw_map)
        self.width = len(self.raw_map[0]) if self.height > 0 else 0
        
        self.start: Optional[Coord] = None
        self.goal: Optional[Coord] = None
        self.obstacles: Set[Coord] = set()
        
        self._parse_map()

    def _parse_map(self):
        for r, line in enumerate(self.raw_map):
            for c, char in enumerate(line):
                if char == 'S':
                    self.start = (r, c)
                elif char == 'G':
                    self.goal = (r, c)
                elif char == '#':
                    self.obstacles.add((r, c))
                    
        if self.start is None:
            raise ValueError(f"Map '{self.name}' missing start symbol 'S'.")
        if self.goal is None:
            raise ValueError(f"Map '{self.name}' missing goal symbol 'G'.")

    def is_valid(self, pos: Coord) -> bool:
        """Checks if a position is within bounds and not an obstacle."""
        r, c = pos
        if 0 <= r < self.height and 0 <= c < self.width:
            return (r, c) not in self.obstacles
        return False

    def get_neighbors(self, pos: Coord) -> List[Tuple[str, Coord]]:
        """
        Returns accessible adjacent cells: Up, Down, Left, Right.
        Action cost is uniform (1).
        """
        r, c = pos
        # Order: Up, Down, Left, Right
        moves = [
            ("Up", (r - 1, c)),
            ("Down", (r + 1, c)),
            ("Left", (r, c - 1)),
            ("Right", (r, c + 1)),
        ]
        valid_neighbors = []
        for action_name, nxt in moves:
            if self.is_valid(nxt):
                valid_neighbors.append((action_name, nxt))
        return valid_neighbors

    def render_path(self, path: List[Coord]) -> str:
        """Overlays the solution path on the ASCII grid using '*' characters."""
        grid = [list(line) for line in self.raw_map]
        path_set = set(path)
        for r, c in path:
            if (r, c) != self.start and (r, c) != self.goal:
                grid[r][c] = '*'
        return "\n".join("".join(row) for row in grid)


# --- Heuristic Functions ---

def manhattan_heuristic(pos: Coord, goal: Coord) -> float:
    """h(n) = |r - r_G| + |c - c_G|"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def euclidean_heuristic(pos: Coord, goal: Coord) -> float:
    """h(n) = sqrt((r - r_G)^2 + (c - c_G)^2)"""
    return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

def zero_heuristic(pos: Coord, goal: Coord) -> float:
    """h(n) = 0 (Equivalent to Uniform Cost Search / Dijkstra)"""
    return 0.0

def scaled_manhattan_heuristic(pos: Coord, goal: Coord, scale: float = 2.0) -> float:
    """h(n) = scale * (|r - r_G| + |c - c_G|)"""
    return scale * (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))


# --- Search Algorithms ---

class SearchResult:
    """Encapsulates the output metrics of a search execution."""
    def __init__(
        self,
        algorithm: str,
        found: bool,
        path: Optional[List[Coord]],
        path_length: int,
        states_expanded: int,
        cost: float
    ):
        self.algorithm = algorithm
        self.found = found
        self.path = path or []
        self.path_length = path_length
        self.states_expanded = states_expanded
        self.cost = cost

    def summary(self) -> str:
        status = "Solution Found" if self.found else "No Solution Found"
        return (
            f"Algorithm: {self.algorithm}\n"
            f"Status: {status}\n"
            f"Path Length: {self.path_length}\n"
            f"Total Cost: {self.cost}\n"
            f"States Expanded: {self.states_expanded}"
        )


def a_star_search(
    env: GridEnvironment,
    heuristic_fn: Callable[[Coord, Coord], float] = manhattan_heuristic,
    heuristic_name: str = "Manhattan"
) -> SearchResult:
    """
    Executes A* Search using evaluation function f(n) = g(n) + h(n).
    Uses a priority queue (min-heap) and closed set.
    """
    start = env.start
    goal = env.goal
    
    # Priority queue stores: (f_score, counter, current_node, g_score)
    # Counter breaks ties deterministically to avoid comparing Coord objects
    counter = 0
    start_h = heuristic_fn(start, goal)
    frontier = [(start_h, counter, start, 0)]
    
    # Tracking optimal g-scores
    g_scores: Dict[Coord, float] = {start: 0.0}
    # Tracking parent pointers for path reconstruction: child -> (parent, action)
    parent_map: Dict[Coord, Optional[Coord]] = {start: None}
    
    closed_set: Set[Coord] = set()
    states_expanded = 0

    while frontier:
        f, _, current, g = heapq.heappop(frontier)

        # Skip if state has already been expanded with a lower or equal g-score
        if current in closed_set:
            continue

        states_expanded += 1
        closed_set.add(current)

        # Goal check upon expansion
        if current == goal:
            # Reconstruct path
            path: List[Coord] = []
            curr: Optional[Coord] = goal
            while curr is not None:
                path.append(curr)
                curr = parent_map[curr]
            path.reverse()
            return SearchResult(
                algorithm=f"A* ({heuristic_name})",
                found=True,
                path=path,
                path_length=len(path) - 1,
                states_expanded=states_expanded,
                cost=g
            )

        # Expand neighbors
        for action_name, neighbor in env.get_neighbors(current):
            tentative_g = g + 1.0  # Unit step cost
            
            if neighbor in closed_set:
                continue

            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                parent_map[neighbor] = current
                h = heuristic_fn(neighbor, goal)
                f_score = tentative_g + h
                counter += 1
                heapq.heappush(frontier, (f_score, counter, neighbor, tentative_g))

    # Queue exhausted without finding goal
    return SearchResult(
        algorithm=f"A* ({heuristic_name})",
        found=False,
        path=None,
        path_length=0,
        states_expanded=states_expanded,
        cost=float('inf')
    )


def bfs_search(env: GridEnvironment) -> SearchResult:
    """
    Executes Breadth-First Search (uninformed blind search) using a FIFO queue.
    Guarantees shortest path in unweighted graphs.
    """
    start = env.start
    goal = env.goal

    if start == goal:
        return SearchResult("BFS", True, [start], 0, 0, 0.0)

    queue = deque([start])
    parent_map: Dict[Coord, Optional[Coord]] = {start: None}
    visited: Set[Coord] = {start}
    states_expanded = 0

    while queue:
        current = queue.popleft()
        states_expanded += 1

        if current == goal:
            path: List[Coord] = []
            curr: Optional[Coord] = goal
            while curr is not None:
                path.append(curr)
                curr = parent_map[curr]
            path.reverse()
            return SearchResult(
                algorithm="BFS",
                found=True,
                path=path,
                path_length=len(path) - 1,
                states_expanded=states_expanded,
                cost=len(path) - 1
            )

        for action_name, neighbor in env.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current
                queue.append(neighbor)

    return SearchResult(
        algorithm="BFS",
        found=False,
        path=None,
        path_length=0,
        states_expanded=states_expanded,
        cost=float('inf')
    )


if __name__ == "__main__":
    env = GridEnvironment(ORIGINAL_WAREHOUSE_MAP, "Original Warehouse")
    res = a_star_search(env)
    print("=" * 60)
    print("A* SEARCH ON ORIGINAL WAREHOUSE:")
    print("=" * 60)
    print(res.summary())
    print("\nSOLUTION PATH VISUALIZATION:")
    print(env.render_path(res.path))
