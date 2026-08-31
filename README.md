# Safe Semantic Planner — PCCST503 Assignment 1

A safe path planner over a finite Cartesian state space, implemented with
**D* Lite**, satisfying all five optimization objectives from the
assignment (reach the goal, never visit a bad state, minimize cost,
maximize distance from bad states, and run within reasonable time), and
supporting efficient replanning when the goal, bad-state set, or
transitions change.

**Language:** Python 3 (course-approved alternative to the suggested C++
interfaces — the class/field names mirror the assignment's C++ structs
one-to-one, see `models.py`).

## Quick Start

```bash
python3 demo.py             # runs & verifies all 6 assignment test cases
python3 benchmark.py        # scalability: incremental vs. cold-start replanning
python3 visualize.py        # matplotlib graph pictures (one window per test case)

pip install streamlit       # one-time
streamlit run app.py        # interactive website — pick a test case, toggle
                             # transitions, change the goal, watch it replan live
```

## Files

| File | Deliverable it satisfies |
|---|---|
| `models.py`, `planner.py` | 1. Source code |
| `DESIGN_REPORT.md` | 2. Design report |
| `run_output.txt`, `benchmark_output.txt` | 3. Experimental results |
| `USER_MANUAL.md` | 4. User manual |
| `demo.py` | 5. Demonstration (all 6 illustrative test cases) |

See `USER_MANUAL.md` to run it and `DESIGN_REPORT.md` for the full write-up
(state representation, data structures, heuristic, safety computation, time
and space complexity, experimental results).
