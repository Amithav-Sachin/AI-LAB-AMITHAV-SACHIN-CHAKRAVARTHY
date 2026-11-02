"""
Algorithm and Heuristic Comparative Benchmark
Laboratory: Search and A* (Tasks 5 and 6)

Performs:
1. Task 5: BFS (Blind Search) vs A* (Informed Search) comparison on Original Warehouse.
2. Task 6: Heuristic ablation and sensitivity analysis:
   - Baseline: Manhattan Distance
   - Variation 1: Zero Heuristic h(n) = 0 (Uniform Cost Search)
   - Variation 2: Euclidean Distance
   - Variation 3: Scaled Heuristic h(n) = 2 * Manhattan (Weighted / Inadmissible)
"""

from search_agent import (
    GridEnvironment,
    a_star_search,
    bfs_search,
    manhattan_heuristic,
    euclidean_heuristic,
    zero_heuristic,
    scaled_manhattan_heuristic,
    ORIGINAL_WAREHOUSE_MAP
)


def run_bfs_vs_astar_comparison():
    print("=" * 70)
    print("TASK 5: COMPARING A* WITH BLIND SEARCH (BFS)")
    print("=" * 70)
    
    # 1. Original Warehouse Map (Constrained Corridor Topology)
    env_orig = GridEnvironment(ORIGINAL_WAREHOUSE_MAP, "Original Warehouse (Corridor)")
    res_bfs_orig = bfs_search(env_orig)
    res_astar_orig = a_star_search(env_orig, manhattan_heuristic, "Manhattan")
    
    print("\n[Scenario A: Original Warehouse (Constrained Corridor - 64 Free Cells)]")
    print(f"{'Measure':<25} | {'BFS (Blind)':<15} | {'A* (Informed)':<15}")
    print("-" * 60)
    print(f"{'Solution Found':<25} | {str(res_bfs_orig.found):<15} | {str(res_astar_orig.found):<15}")
    print(f"{'Path Length':<25} | {res_bfs_orig.path_length:<15} | {res_astar_orig.path_length:<15}")
    print(f"{'States Expanded':<25} | {res_bfs_orig.states_expanded:<15} | {res_astar_orig.states_expanded:<15}")
    print("-" * 60)
    print("Key Finding: In a single winding corridor with 64 total free cells and no")
    print("alternative branches, both algorithms must traverse the corridor, expanding 64 states.")

    # 2. Branching Open Warehouse Map (Illustrating Heuristic Pruning)
    open_map = """#################
#S..............#
#...............#
#......#####....#
#......#...#....#
#......#...#....#
#......#####....#
#..............G#
#################"""
    env_open = GridEnvironment(open_map, "Open Warehouse with Obstacle")
    res_bfs_open = bfs_search(env_open)
    res_astar_open = a_star_search(env_open, manhattan_heuristic, "Manhattan")
    
    print("\n[Scenario B: Branching/Open Warehouse (Branching Factor > 1)]")
    print(f"{'Measure':<25} | {'BFS (Blind)':<15} | {'A* (Informed)':<15}")
    print("-" * 60)
    print(f"{'Solution Found':<25} | {str(res_bfs_open.found):<15} | {str(res_astar_open.found):<15}")
    print(f"{'Path Length':<25} | {res_bfs_open.path_length:<15} | {res_astar_open.path_length:<15}")
    print(f"{'States Expanded':<25} | {res_bfs_open.states_expanded:<15} | {res_astar_open.states_expanded:<15}")
    print("-" * 60)
    print(f"Heuristic Impact: A* reduces state expansions by {res_bfs_open.states_expanded - res_astar_open.states_expanded} states "
          f"({((res_bfs_open.states_expanded - res_astar_open.states_expanded) / res_bfs_open.states_expanded)*100:.1f}% reduction)!")
    print("=" * 70)
    return res_bfs_orig, res_astar_orig


def run_heuristic_investigation():
    print("\n" + "=" * 70)
    print("TASK 6: HEURISTIC SENSITIVITY INVESTIGATION")
    print("=" * 70)
    
    configs = [
        ("Manhattan (Baseline)", manhattan_heuristic),
        ("Zero Heuristic h(n)=0", zero_heuristic),
        ("Euclidean Distance", euclidean_heuristic),
        ("Scaled 2*Manhattan", scaled_manhattan_heuristic),
    ]

    # Benchmark on Original Map
    env_orig = GridEnvironment(ORIGINAL_WAREHOUSE_MAP, "Original Warehouse")
    print("[Scenario A: Original Warehouse Benchmark]")
    print(f"{'Heuristic Function':<25} | {'Found?':<8} | {'Path Length':<12} | {'States Expanded':<16} | {'Admissible?':<12}")
    print("-" * 80)
    for name, fn in configs:
        res = a_star_search(env_orig, fn, name)
        admissible = "No (Overest)" if "Scaled" in name else "Yes"
        print(f"{name:<25} | {str(res.found):<8} | {res.path_length:<12} | {res.states_expanded:<16} | {admissible:<12}")

    # Benchmark on Branching Map to reveal pruning differences
    open_map = """#################
#S..............#
#...............#
#......#####....#
#......#...#....#
#......#...#....#
#......#####....#
#..............G#
#################"""
    env_open = GridEnvironment(open_map, "Branching Warehouse")
    print("\n[Scenario B: Branching Open Warehouse Benchmark]")
    print(f"{'Heuristic Function':<25} | {'Found?':<8} | {'Path Length':<12} | {'States Expanded':<16} | {'Admissible?':<12}")
    print("-" * 80)
    results = []
    for name, fn in configs:
        res = a_star_search(env_open, fn, name)
        admissible = "No (Overest)" if "Scaled" in name else "Yes"
        print(f"{name:<25} | {str(res.found):<8} | {res.path_length:<12} | {res.states_expanded:<16} | {admissible:<12}")
        results.append((name, res, admissible))
    print("=" * 80)
    return results


def run_all_comparisons():
    run_bfs_vs_astar_comparison()
    run_heuristic_investigation()


if __name__ == "__main__":
    run_all_comparisons()
