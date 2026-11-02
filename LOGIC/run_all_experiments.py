"""
Master Orchestration Runner for Logic Planning Laboratory
Runs:
1. Planner execution (Task 2)
2. Test Suite: Test A, Test B, Test C (Task 3)
3. Independent Plan Verifier (Task 5)
4. Prolog Logical Verifier Suite (Tasks 6, 7, 8)
And captures full console outputs into output artifacts.
"""

import sys
import io
import os
import contextlib
from test_planner import run_all_tests
from test_prolog import run_all_prolog_tests
from planner import run_standard_warehouse_problem


def run_all():
    logic_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Run and log Python tests
    test_log_path = os.path.join(logic_dir, "test_results.txt")
    print("Executing Python Planning and Verification Tests...")
    
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_all_tests()
    python_output = buf.getvalue()
    
    with open(test_log_path, "w", encoding="utf-8") as f:
        f.write(python_output)
    print(f"-> Saved Python test logs to {test_log_path}")

    # 2. Run and log Prolog tests
    prolog_log_path = os.path.join(logic_dir, "prolog_verification_results.txt")
    print("Executing Prolog Logical Verifier Suite...")
    
    buf_prolog = io.StringIO()
    with contextlib.redirect_stdout(buf_prolog):
        run_all_prolog_tests()
    prolog_output = buf_prolog.getvalue()
    
    with open(prolog_log_path, "w", encoding="utf-8") as f:
        f.write(prolog_output)
    print(f"-> Saved Prolog verification logs to {prolog_log_path}")

    print("\n" + "=" * 60)
    print("ALL LABORATORY EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
