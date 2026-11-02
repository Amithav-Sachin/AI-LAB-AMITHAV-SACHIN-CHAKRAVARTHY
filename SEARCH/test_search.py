"""
Systematic Test Suite for Search Agent
Laboratory: Search and A* (Task 3)

Implements:
- Test 1: Original warehouse
- Test 2: Trivial case (immediate neighbor)
- Test 3: No solution (inaccessible goal)
- Test 4: Alternative paths (multiple paths of different lengths)
"""

from search_agent import (
    GridEnvironment,
    a_star_search,
    bfs_search,
    ORIGINAL_WAREHOUSE_MAP
)

TEST_2_TRIVIAL_MAP = """#####
#SG##
#####"""

TEST_3_NO_SOLUTION_MAP = """#######
#S....#
###.###
#...#G#
#######"""

TEST_4_ALTERNATIVE_PATHS_MAP = """#########
#S......#
#.#####.#
#.......G
#########"""
# In this map, path along top vs path along bottom or split corridors:
# Let's design a clear asymmetric split:
# Corridor 1 (Short path above obstacle, length = 6):
# Corridor 2 (Long path detour below obstacle, length = 10):
TEST_4_MAP = """###########
#S........#
#.#######.#
#.......#G#
###########"""


def run_test_1():
    print("\n" + "=" * 70)
    print("RUNNING TEST 1: ORIGINAL WAREHOUSE BENCHMARK")
    print("=" * 70)
    env = GridEnvironment(ORIGINAL_WAREHOUSE_MAP, "Original Warehouse")
    res = a_star_search(env)
    
    print(res.summary())
    print("\nVisualized Solution Trajectory:")
    print(env.render_path(res.path))
    
    assert res.found is True, "Test 1 Failed: No path found!"
    assert res.path_length > 0, "Test 1 Failed: Path length must be > 0"
    assert res.path[0] == env.start, "Path must start at S"
    assert res.path[-1] == env.goal, "Path must terminate at G"
    return res


def run_test_2():
    print("\n" + "=" * 70)
    print("RUNNING TEST 2: TRIVIAL CASE (Immediately Adjacent Goal)")
    print("=" * 70)
    env = GridEnvironment(TEST_2_TRIVIAL_MAP, "Trivial Map")
    res = a_star_search(env)
    
    print(res.summary())
    print("\nVisualized Path:")
    print(env.render_path(res.path))
    
    assert res.found is True, "Test 2 Failed: No path found!"
    assert res.path_length == 1, f"Test 2 Failed: Expected path length 1, got {res.path_length}"
    assert res.path == [env.start, env.goal], "Path should be directly from start to goal!"
    print("-> Verified: Successfully found one-step solution immediately.")
    return res


def run_test_3():
    print("\n" + "=" * 70)
    print("RUNNING TEST 3: NO SOLUTION (Completely Inaccessible Goal)")
    print("=" * 70)
    env = GridEnvironment(TEST_3_NO_SOLUTION_MAP, "Blocked Map")
    res = a_star_search(env)
    
    print(res.summary())
    assert res.found is False, "Test 3 Failed: Claimed to find path when goal is inaccessible!"
    assert res.path == [], "Path should be empty for unsolvable case"
    print("-> Verified: Successfully terminated and reported failure without looping.")
    return res


def run_test_4():
    print("\n" + "=" * 70)
    print("RUNNING TEST 4: ALTERNATIVE PATHS (Short vs Long Route)")
    print("=" * 70)
    # Map with two distinct routes:
    # A short path directly above the obstacle and a longer roundabout detour below
    alt_map = """###########
#S.......G#
#.#######.#
#.........#
###########"""
    env = GridEnvironment(alt_map, "Alternative Paths Map")
    res_astar = a_star_search(env)
    res_bfs = bfs_search(env)
    
    print("A* Result:")
    print(res_astar.summary())
    print("\nA* Path:")
    print(env.render_path(res_astar.path))
    
    print("\nBFS Ground Truth Result:")
    print(res_bfs.summary())
    
    assert res_astar.found is True, "Test 4 Failed: No path found"
    assert res_astar.path_length == res_bfs.path_length, (
        f"A* returned sub-optimal path: length {res_astar.path_length} vs BFS {res_bfs.path_length}"
    )
    print(f"-> Verified: Returned optimal shortest path of length {res_astar.path_length}.")
    return res_astar


def run_all_tests():
    t1 = run_test_1()
    t2 = run_test_2()
    t3 = run_test_3()
    t4 = run_test_4()
    
    print("\n" + "=" * 75)
    print("SUMMARY MATRIX OF SYSTEMATIC TESTS (TASK 3)")
    print("=" * 75)
    print(f"{'Test':<15} | {'Solution Found?':<16} | {'Path Length':<12} | {'States Expanded':<16}")
    print("-" * 75)
    print(f"{'Test 1 (Orig)':<15} | {str(t1.found):<16} | {t1.path_length:<12} | {t1.states_expanded:<16}")
    print(f"{'Test 2 (Triv)':<15} | {str(t2.found):<16} | {t2.path_length:<12} | {t2.states_expanded:<16}")
    print(f"{'Test 3 (None)':<15} | {str(t3.found):<16} | {'N/A':<12} | {t3.states_expanded:<16}")
    print(f"{'Test 4 (Alt)':<15} | {str(t4.found):<16} | {t4.path_length:<12} | {t4.states_expanded:<16}")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()
