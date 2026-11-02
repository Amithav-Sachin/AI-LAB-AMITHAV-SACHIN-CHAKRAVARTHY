"""
Automated Test Runner for Prolog Logical Verifier
Laboratory: Logical Reasoning for Planning (Tasks 6, 7 & 8)

Interacts directly with SWI-Prolog ('swipl') to run and verify:
- Task 6 queries: can_move(a,b), can_move(a,c)
- Task 7 queries: valid_move(a,b), valid_move(b,c), valid_move(a,c), valid_move_plan(...)
- Task 8 query: reduce_speed
"""

import subprocess
import os

def run_prolog_query(pl_file: str, query: str) -> bool:
    """Executes a Prolog query via SWI-Prolog CLI and checks truth value."""
    goal = f"({query} -> writeln('RESULT:TRUE') ; writeln('RESULT:FALSE'))."
    cmd = ["swipl", "-s", pl_file, "-g", goal, "-t", "halt"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SWI-Prolog error: {proc.stderr}")
    return "RESULT:TRUE" in proc.stdout

def run_all_prolog_tests():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    planner_pl = os.path.join(script_dir, "planner.pl")
    driving_pl = os.path.join(script_dir, "driving_reasoning.pl")

    print("=" * 70)
    print("PROLOG LOGICAL VERIFICATION TEST SUITE (TASKS 6, 7, 8)")
    print("=" * 70)

    # Task 6
    print("\n--- Task 6: Basic Movement Verification ---")
    t6_q1 = run_prolog_query(planner_pl, "can_move(a,b)")
    t6_q2 = run_prolog_query(planner_pl, "can_move(a,c)")
    print(f"?- can_move(a,b).  ==> {'true.' if t6_q1 else 'false.'}")
    print(f"?- can_move(a,c).  ==> {'true.' if t6_q2 else 'false.'}")
    assert t6_q1 is True, "can_move(a,b) should succeed!"
    assert t6_q2 is False, "can_move(a,c) should fail!"

    # Task 7
    print("\n--- Task 7: Proposed Plan Action Verification ---")
    t7_q1 = run_prolog_query(planner_pl, "valid_move(a,b)")
    t7_q2 = run_prolog_query(planner_pl, "valid_move(b,c)")
    t7_q3 = run_prolog_query(planner_pl, "valid_move(a,c)")
    print(f"?- valid_move(a,b). ==> {'true.' if t7_q1 else 'false.'}")
    print(f"?- valid_move(b,c). ==> {'true.' if t7_q2 else 'false.'}")
    print(f"?- valid_move(a,c). ==> {'true.' if t7_q3 else 'false.'}")
    assert t7_q1 is True and t7_q2 is True, "Valid moves must succeed!"
    assert t7_q3 is False, "Invalid direct move a->c must fail!"

    print("\n--- Task 7 Extension: Full Plan Trajectory Check ---")
    t7_plan_valid = run_prolog_query(planner_pl, "valid_move_plan([move(a,b), move(b,c)])")
    t7_plan_invalid = run_prolog_query(planner_pl, "valid_move_plan([move(a,c)])")
    print(f"?- valid_move_plan([move(a,b), move(b,c)]). ==> {'true.' if t7_plan_valid else 'false.'}")
    print(f"?- valid_move_plan([move(a,c)]).              ==> {'true.' if t7_plan_invalid else 'false.'}")
    assert t7_plan_valid is True
    assert t7_plan_invalid is False

    # Task 8
    print("\n--- Task 8: Logical Reasoning Chains (Deductive Modus Ponens) ---")
    t8_q = run_prolog_query(driving_pl, "reduce_speed")
    print(f"?- reduce_speed.   ==> {'true.' if t8_q else 'false.'}")
    assert t8_q is True, "reduce_speed query should succeed by deductive inference!"

    print("\n" + "=" * 70)
    print("ALL PROLOG VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_prolog_tests()
