"""
Master Orchestration Runner for Agents Laboratory
Executes:
1. Standard warehouse benchmark (A* vs BFS)
2. Blocked map unreachable goal handling
3. Plan execution simulation
And logs verbatim console outputs.
"""

import os
import io
import contextlib
from test_agent import run_benchmark


def run_all():
    agents_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(agents_dir, "agent_execution_results.txt")
    
    print("Running Agents Laboratory Benchmark Suite...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_benchmark()
    output = buf.getvalue()
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"-> Saved execution logs to {log_path}")
    print("\n" + "=" * 60)
    print("ALL AGENT EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
