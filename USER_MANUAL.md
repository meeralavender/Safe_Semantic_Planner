# User Manual — Safe Semantic Planner

## 1. Requirements

* Python 3.9+ (the core planner uses only the standard library: `heapq`,
  `dataclasses`, `itertools`, `time`, `tracemalloc`, `math`, `random`)
* `matplotlib` — **only** needed for `visualize.py`. Install with:
  ```bash
  pip install matplotlib
  ```
  Everything else (`demo.py`, `benchmark.py`, the planner itself) runs
  with no installs at all.

## 2. Project Layout

```
safe_planner/
├── models.py          State, Transition, PlanningProblem, PlanningResult
├── planner.py          DStarLitePlanner (the algorithm) + PlannerWeights
├── scenarios.py         Builders for the 6 assignment test-case graphs
├── demo.py              Runs & asserts the 6 assignment test cases
├── benchmark.py         Scalability benchmark (incremental vs cold replan)
├── visualize.py          Draws each test case with matplotlib
├── app.py                 Streamlit website (interactive, run with `streamlit run app.py`)
├── visuals/               PNG output of visualize.py --save (pre-generated)
├── DESIGN_REPORT.md   Design report (state rep., complexity, safety design)
├── USER_MANUAL.md     This file
└── run_output.txt, benchmark_output.txt   Captured experimental output
```

## 3. Running the Test Cases

```bash
cd safe_planner
python3 demo.py
```

This prints, for each of the six assignment test cases: success/failure,
the resulting path, total cost, average safety score, minimum distance to
a bad state, number of states explored, planning time, and peak memory.

## 4. Running the Scalability Benchmark

```bash
python3 benchmark.py
```

Generates random Cartesian graphs (200/500/1000 states) and reports how
much faster an incremental D* Lite replan is compared to rebuilding the
planner from scratch after a single transition is disabled.

## 5. Running It as a Website (Streamlit)

`app.py` turns the whole project into a local interactive website — pick a
test case from a sidebar, watch the graph render live, and actually
interact with the dynamic scenarios (toggle a transition off, change the
goal, add a shortcut) to see the planner replan in your browser instead of
reading terminal text.

Install Streamlit once:

```bash
pip install streamlit
```

Then, from inside the `safe_planner` folder:

```bash
streamlit run app.py
```

This prints a local URL (usually `http://localhost:8501`) and should open
it in your default browser automatically. If it doesn't, just paste that
URL into your browser manually. To stop the server, go back to the
terminal and press `Ctrl+C`.

The sidebar lets you switch between all 6 test cases plus a "Scalability
Benchmark" page where you can pick a graph size and click a button to
compare incremental vs. cold-start replanning times live.

## 7. Visualizing the Graphs (matplotlib, static images)

`visualize.py` draws each test case's graph: states as points, bad states
as red X markers, the start as a green circle, the goal as a gold star, and
the planner's chosen path highlighted as a thick blue arrow trail. Requires
`matplotlib` (`pip install matplotlib` if you don't have it — everything
else in this project is standard library only).

```bash
python3 visualize.py            # opens a matplotlib window per test case
python3 visualize.py --save     # saves PNGs to visuals/ instead
python3 visualize.py --case 2   # only show/save Test Case 2
```

In VS Code, `python3 visualize.py` pops up a native plot window (via
whatever GUI backend matplotlib picks on your OS) for Test Case 1; closing
that window advances to the next test case's window, and so on through all
6. If you're running somewhere without a display (e.g. an SSH session or a
container), use `--save` instead — the PNGs land in `visuals/` and can be
opened like any image file.

## 8. Using the Planner in Your Own Code

```python
from models import State, Transition, PlanningProblem
from planner import DStarLitePlanner, PlannerWeights

# 1. Define states (each a d-dimensional embedding)
states = [
    State(0, (0.0, 0.0)),
    State(1, (1.0, 0.0)),
    State(2, (2.0, 0.0)),
]

# 2. Define directed transitions
transitions = [
    Transition(id=0, src=0, dst=1, cost=1.0, safety=0.95, reliability=0.9),
    Transition(id=1, src=1, dst=2, cost=1.0, safety=0.95, reliability=0.9),
]

# 3. Build the problem (bad_states is optional)
problem = PlanningProblem.build(
    states, transitions, initial_state=0, goal_state=2, bad_states=[],
)

# 4. Optionally tune how cost/safety/reliability/bad-proximity are weighed
weights = PlannerWeights(
    w_cost=1.0, w_unsafety=2.0, w_unreliability=1.0,
    w_bad_proximity=3.0, safety_radius=2.0,
)

# 5. Plan
planner = DStarLitePlanner(problem, weights)
result = planner.plan()

print(result.success, result.state_path, result.total_cost)
```

## 9. Handling a Dynamic Environment

All of these return a fresh `PlanningResult` after an efficient incremental
replan (see DESIGN_REPORT.md §5 for the complexity of each):

```python
# A transition becomes unavailable / available again
result = planner.set_transition_availability(transition_id=1, available=False)

# A transition's cost changes (e.g. traffic, congestion, re-measured cost)
result = planner.set_transition_cost(transition_id=0, cost=5.0)

# A brand-new transition appears (e.g. a discovered shortcut)
new_edge = Transition(id=99, src=0, dst=2, cost=1.2, safety=0.98, reliability=1.0)
result = planner.add_transition(new_edge)

# A transition is permanently removed
result = planner.remove_transition(transition_id=1)

# The set of bad/forbidden states changes
result = planner.update_bad_states([3, 7, 12])

# The goal changes (more expensive than the updates above — see report §5)
result = planner.update_goal(new_goal=5)

# The agent's current position changes (cheap, classic D* Lite start-move)
planner.update_start(new_start=1)
```

## 10. Interpreting `PlanningResult`

| Field | Meaning |
|---|---|
| `success` | `True` if a path to the goal exists that avoids all bad states |
| `state_path` | Ordered list of state ids from `initial_state` to `goal_state` |
| `transition_path` | Ordered list of transition ids taken |
| `total_cost` | Sum of the raw `cost` field along the path (not the internal search weight) |
| `safety_score` | Average of the per-transition `safety` field along the path |
| `min_bad_distance` | Minimum Euclidean distance, over all visited states, to the nearest bad state |
| `states_explored` | Number of vertex expansions performed by this `plan()`/incremental call — a proxy for search effort |
| `planning_time_s` | Wall-clock time for this specific call, in seconds |
| `peak_memory_kb` | Peak additional memory (via `tracemalloc`) used during this call |

## 11. Tuning `PlannerWeights`

* Increase `w_cost` relative to the others to prioritize cheap paths.
* Increase `w_unsafety` / `w_unreliability` to avoid transitions with low
  per-edge `safety`/`reliability` scores.
* Increase `w_bad_proximity` and/or `safety_radius` to push the path
  farther away from bad states, at the cost of higher total cost (see
  Test Case 3 in `demo.py` / DESIGN_REPORT.md §7 for a worked example).

## 12. Known Limitations

* The safety-distance objective is optimized approximately via a per-edge
  penalty, not exactly as a global bottleneck (see DESIGN_REPORT.md §4).
* `update_goal()` is not as cheap as the other incremental updates, by
  design — this mirrors a genuine limitation of D* Lite's backward-search
  formulation, discussed in the design report.
