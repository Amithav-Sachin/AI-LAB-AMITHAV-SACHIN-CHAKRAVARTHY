"""
Master Orchestration Runner for Search Laboratory
Executes:
1. Systematic test suite: Tests 1, 2, 3, 4 (Task 3)
2. Algorithm comparative benchmarks: BFS vs A* (Task 5)
3. Heuristic sensitivity ablations: Zero, Manhattan, Euclidean, Scaled (Task 6)
And logs full verbatim console outputs.
"""

import os
import io
import contextlib
from test_search import run_all_tests
from compare_algorithms import run_all_comparisons


def run_all():
    search_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Run and log Task 3 tests
    test_log_path = os.path.join(search_dir, "test_results.txt")
    print("Running Task 3 Test Suite...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_all_tests()
    test_output = buf.getvalue()
    with open(test_log_path, "w", encoding="utf-8") as f:
        f.write(test_output)
    print(f"-> Saved test logs to {test_log_path}")

    # 2. Run and log Task 5 & Task 6 comparisons
    comp_log_path = os.path.join(search_dir, "heuristic_comparison_results.txt")
    print("Running Tasks 5 & 6 Algorithm and Heuristic Comparisons...")
    buf_comp = io.StringIO()
    with contextlib.redirect_stdout(buf_comp):
        run_all_comparisons()
    comp_output = buf_comp.getvalue()
    with open(comp_log_path, "w", encoding="utf-8") as f:
        f.write(comp_output)
    print(f"-> Saved comparison logs to {comp_log_path}")

    print("\n" + "=" * 60)
    print("ALL SEARCH LABORATORY EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
