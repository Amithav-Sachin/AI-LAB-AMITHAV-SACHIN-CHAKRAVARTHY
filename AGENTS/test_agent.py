"""
Test Suite and Empirical Evaluation for Goal-Based Agent
Laboratory: Agents - Constructing a Goal-Based Agent using an LLM
"""

from goal_based_agent import WarehouseEnvironment, GoalBasedAgent, WAREHOUSE_MAP

BLOCKED_MAP = """#####################
#S....#...........#G#
#.##....##########.##
#....##............##
#.######.###.#.###.##
#........#.........##
#####################"""


def run_benchmark():
    print("=" * 70)
    print("TEST 1: STANDARD WAREHOUSE NAVIGATION (A* vs BFS)")
    print("=" * 70)
    env = WarehouseEnvironment(WAREHOUSE_MAP)
    
    agent_astar = GoalBasedAgent(env, algorithm="A*")
    success_astar = agent_astar.formulate_plan()
    
    agent_bfs = GoalBasedAgent(env, algorithm="BFS")
    success_bfs = agent_bfs.formulate_plan()
    
    print(f"{'Algorithm':<15} | {'Plan Found?':<12} | {'Path Length':<12} | {'Nodes Expanded':<16}")
    print("-" * 60)
    print(f"{'A* (Manhattan)':<15} | {str(success_astar):<12} | {len(agent_astar.plan)-1:<12} | {agent_astar.nodes_expanded:<16}")
    print(f"{'BFS (Blind)':<15} | {str(success_bfs):<12} | {len(agent_bfs.plan)-1:<12} | {agent_bfs.nodes_expanded:<16}")
    
    assert success_astar and success_bfs, "Both algorithms must find a path!"
    assert len(agent_astar.plan) == len(agent_bfs.plan), "Both should find an optimal path!"
    print(f"\nVisualized Trajectory (* = Path):")
    print(env.render(path=agent_astar.plan))

    print("\n" + "=" * 70)
    print("TEST 2: UNREACHABLE GOAL (Blocked Map Handling)")
    print("=" * 70)
    blocked_env = WarehouseEnvironment(BLOCKED_MAP)
    blocked_agent = GoalBasedAgent(blocked_env, algorithm="A*")
    blocked_success = blocked_agent.formulate_plan()
    
    print(f"Goal Found: {blocked_success}")
    print(f"Nodes Expanded Before Exhaustion: {blocked_agent.nodes_expanded}")
    assert not blocked_success, "Agent must report failure for unreachable goal!"
    print("-> Verified: Successfully reported no path exists without infinite loops.")

    print("\n" + "=" * 70)
    print("TEST 3: STEP-BY-STEP PLAN EXECUTION")
    print("=" * 70)
    trace = agent_astar.execute_plan()
    print(f"Executed {len(trace)-1} steps successfully.")
    print(f"Final Agent Position: {agent_astar.current_pos} (Target: {env.goal})")
    assert agent_astar.current_pos == env.goal, "Agent failed to reach target position!"
    print("-> Verified: Execution reached delivery bay G successfully.")


if __name__ == "__main__":
    run_benchmark()
