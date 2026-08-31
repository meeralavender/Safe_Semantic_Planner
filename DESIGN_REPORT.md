# Design Report — Safe Semantic Planner in a Finite Cartesian State Space

**Course:** PCCST503 — Machine Learning
**Assignment 1**
**Algorithm chosen:** D* Lite (incremental replanning variant of LPA*/A*)
**Implementation language:** Python 3 (per course guidance: C++ preferred for
production builds, Python accepted for the assignment)

---

## 1. State Representation

Each state `s_i = (x_1, ..., x_d)` is represented by the `State` dataclass
(`models.py`):

```python
@dataclass
class State:
    id: int
    embedding: Tuple[float, ...]
```

The embedding is a `d`-dimensional tuple, so the same class transparently
supports 2‑D toy examples, higher-dimensional feature spaces, or embeddings
pulled from a knowledge graph (see §8, bonus). `State.distance_to()`
computes Euclidean distance in `R^d`, which is used for two purposes:

1. Computing each state's distance to the nearest bad state (§4).
2. Providing a consistent search heuristic (§3).

A `Transition` mirrors the C++ struct in the assignment exactly:

```python
@dataclass
class Transition:
    id: int
    src: int
    dst: int
    cost: float
    safety: float        # in [0,1]
    reliability: float    # in [0,1]
    available: bool = True
```

`PlanningProblem` and `PlanningResult` are direct translations of the
suggested `PlanningProblem` / `PlanningResult` C++ interfaces.

## 2. Data Structures

| Structure | Purpose | Complexity |
|---|---|---|
| `states: Dict[int, State]` | O(1) state lookup by id | O(V) space |
| `out_edges`, `in_edges: Dict[int, List[Transition]]` | Adjacency lists, forward and reverse | O(V+E) space |
| `g, rhs: Dict[int, float]` | D* Lite's two cost estimates per state | O(V) space |
| `open_heap: List[(key, seq, id)]` (binary heap via `heapq`) | Priority queue ordered by D* Lite key | O(log V) push/pop |
| `open_set: Dict[int, key]` | Tracks each vertex's *current* key so stale heap entries can be lazily discarded | O(1) membership check |
| `_bad_dist: Dict[int, float]` | Precomputed distance from every state to the nearest bad state | O(V·k) to build (k = #bad states), O(1) lookup |

The reverse adjacency list (`in_edges`) is what makes D* Lite's backward
search from the goal efficient: `update_vertex` needs a state's
*predecessors* every time its `g`-value changes, which would be O(E) per
update on an edge list but is O(deg⁻(s)) with `in_edges`.

Bad states are **not deleted** from the state dictionary (their embeddings
are still needed for the distance computation in §4); instead they are
excluded structurally: `successors()` and `predecessors()` filter out any
transition whose source or destination is a bad state, and unavailable
transitions are filtered the same way. This guarantees objective 2 ("never
visit a bad state") holds by construction — the search can never expand
through a bad state, not merely "prefers not to".

## 3. Heuristic Function

D* Lite requires an admissible, consistent heuristic `h(start, s)` to stay
efficient (without one it degrades to Dijkstra). Because the assignment
allows `cost` to be an arbitrary per-edge scalar — not necessarily equal to
Euclidean distance — using raw Euclidean distance as `h` is **not**
guaranteed admissible.

The implementation instead computes a *scale factor*:

```
scale = min over all available edges (u,v) of  weight(u,v) / euclidean_dist(u,v)
```

and sets `h(start, s) = scale * euclidean_dist(start, s)`. Since `scale` is
a global lower bound on "weight per unit embedding-space distance", `h`
never overestimates the true shortest-path weight to any state, which
preserves consistency (`h(s) <= weight(s,s') + h(s')` for every edge). If no
finite scale can be derived (e.g., a degenerate graph with all zero-length
edges), the implementation safely falls back to `h = 0`, which reduces the
search to uninformed Dijkstra — always correct, just less directed.

## 4. Safety Computation

The assignment's objective function is

```
Score(P) = alpha*G - beta*C + gamma*D + delta*R
```

`G` (goal completion) and `D` (minimum distance to any bad state along the
path) are **not** additive over edges — `D` in particular is a bottleneck
(min) over the whole path, which classic single-source shortest-path
algorithms cannot optimize directly. `PlannerWeights` turns this
multi-objective score into a single additive edge weight D* Lite *can*
optimize, as a practical approximation:

```
weight(u,v) = w_cost * cost(u,v)
            + w_unsafety * (1 - safety(u,v))
            + w_unreliability * (1 - reliability(u,v))
            + w_bad_proximity * max(0, safety_radius - dist(v, nearest_bad))
```

`dist(v, nearest_bad)` is the Euclidean distance from the destination state
to the closest bad state, precomputed once per graph version in
`_precompute_bad_distances()` (O(V·k)). Any edge that steps into the
"danger radius" around a bad state accrues a proportional penalty, so the
search is steered away from bad states even when a shorter, riskier route
exists — this is what produces the Test Case 3 behaviour (§7): a cost-only
weighting picks the cheap path close to the bad state, while a
safety-weighted configuration pays extra cost for a larger safety margin.

**Limitation and exact alternative.** Because the penalty is per-edge, the
final path's true bottleneck safety margin is only *approximately*
maximized, not provably optimal in the max‑min sense of objective 4. An
exact solution to "maximize the minimum distance to a bad state, then
minimize cost" is the classic **widest-path / bottleneck-shortest-path**
problem, solvable with a modified Dijkstra that keeps the *minimum* edge
margin seen so far instead of the *sum* of costs. This is noted as a
natural extension (see `benchmark.py`'s structure, which could be adapted)
rather than implemented as the primary planner, since it would require a
second, non-additive search that cannot share the incremental D* Lite
machinery used for the dynamic-environment requirement.

## 5. Time Complexity

* **Cold start** (`compute_shortest_path` from an empty `g/rhs` table): each
  vertex is expanded at most a small constant number of times before its
  `g` value converges (standard D* Lite/LPA* result), each expansion does
  O(deg⁻(s)) predecessor updates, each update is an O(log V) heap
  operation. Overall: **O(E log V)** — the same bound as Dijkstra/A* with a
  binary heap.
* **Incremental replan after a local change** (single edge cost/
  availability flip, single edge insertion/removal): only vertices whose
  shortest-path estimate is actually affected are re-inserted into the open
  list. In the common case where the change is far from the currently
  optimal path, this is **O(k log V)** for a small, localized `k` ≪ V — this
  is the entire reason D* Lite is preferred over recomputing from scratch
  for the "dynamic environment" part of the assignment (Test Cases 4 and
  6). The benchmark in §7 measures this empirically.
* **Bad-state set change** (`update_bad_states`): requires recomputing
  `_bad_dist` for every state (O(V·k)) and re-validating every vertex's
  `rhs` (O(V) `update_vertex` calls, each O(deg⁻) ), so this is the most
  expensive incremental update short of a goal change — still far cheaper
  than discarding and rebuilding the adjacency lists.
* **Goal change** (`update_goal`): D* Lite's backward search is rooted at
  the goal, so changing it invalidates essentially all `g/rhs` values.
  The implementation re-initializes the search state (`g, rhs, open list`)
  but *keeps* the graph, adjacency lists and precomputed bad-state
  distances, then reruns `compute_shortest_path`, i.e. **O(E log V)**, same
  as a cold start of the search (but not of graph construction). This
  matches the assignment's requirement to replan "without rebuilding all
  data structures whenever possible" — only the search-state structures are
  rebuilt, not the whole planner.

## 6. Space Complexity

`O(V + E)`: adjacency lists are `O(V + E)`; `g`, `rhs`, `_bad_dist` and the
open-set index are each `O(V)`; the open heap holds at most `O(V)` live
entries (stale lazily-deleted entries are bounded by the number of
`update_vertex` calls made so far, so in the worst case the heap is
`O(V + updates)`, still linear in the total work done).

## 7. Experimental Results

All six illustrative test cases from the assignment were implemented in
`demo.py` and pass. Summary (full trace in `run_output.txt`):

| Test case | Result | Path | Cost | States explored | Time (ms) |
|---|---|---|---|---|---|
| 1. Basic reachability | pass | S→A→B→G | 3.00 | 4 | 0.09 |
| 2. Bad state avoidance | pass | S→C→D→G (avoids X) | 4.50 | 4 | 0.06 |
| 3a. Safety margin, cost-only weights | pass | close-but-cheap path | 3.00 (min-dist-to-bad 0.80) | 4 | 0.09 |
| 3b. Safety margin, safety-weighted | pass | far-but-costlier path | 6.00 (min-dist-to-bad 1.41) | 5 | 0.07 |
| 4. Dynamic transition (A,G) removed | pass | S→A→G, then S→C→D→G | 2.00 → 4.50 | 3 → 8 | 0.05 → 0.11 |
| 5. Goal update | pass | goal 3, then goal 2 | 3.00 → 2.00 | 4 → 7 | 0.05 → 0.03 |
| 6. Transition addition (shortcut) | pass | S→A→B→G, then S→G | 3.00 → 1.20 | 4 → 5 | 0.04 → 0.01 |

Test Case 3 is the clearest illustration of the safety/cost trade-off: with
`w_bad_proximity = 0` the planner is indifferent to danger and takes the
cheap path 0.80 units from the bad state; raising `w_bad_proximity` (and
`w_unsafety`) makes the planner accept double the cost for a path that
stays 1.41 units away.

### Scalability benchmark (`benchmark.py`)

Random Cartesian graphs of increasing size, comparing an **incremental**
replan (flip one transition's availability) against a **cold-start** replan
on an equivalent fresh planner:

| States | Edges | Cold replan (ms) | Incremental replan (ms) | Speedup | Explored (cold) | Explored (incr.) |
|---|---|---|---|---|---|---|
| 50 | 300 | 4.05 | 1.68 | 2.4x | 28 | 32 |
| 200 | 800 | 16.28 | 8.97 | 1.8x | 176 | 232 |
| 500 | 2500 | 14.39 | 13.01 | 1.1x | 96 | 100 |
| 1000 | 6000 | 42.53 | 39.35 | 1.1x | 200 | 214 |

The speedup is largest on smaller/denser graphs and shrinks as the flipped
edge sits farther from the goal-rooted search frontier — consistent with
the theoretical result that incremental D* Lite wins precisely when a
change is *local* to a small region of the already-explored search space,
not uniformly across all graph sizes. (Random graphs place the flipped edge
at a random point on the current best path, which is a conservative,
not best-case, test of the incremental benefit.)

## 8. Notes on the Bonus Objectives

* **Dynamic environment (implemented as core, not bonus):** goal changes,
  bad-state set changes, transition availability/cost changes, and
  transition insertion/removal are all supported without rebuilding the
  graph from scratch (`planner.py`, §5 above).
* **Knowledge-graph test (bonus, suggested extension):** because `State`
  only requires a `d`-dimensional embedding, any knowledge-graph embedding
  (e.g. TransE/Node2Vec vectors) can be dropped in directly as `embedding`,
  and graph edges/relations become `Transition`s with `cost` derived from
  edge weight or relation confidence. This was not run against a real KG
  dataset for this submission due to time constraints, but the interface
  requires no code changes to support it.
* **Multi-goal planning / learning-based heuristic** were considered but
  intentionally left out of scope to keep the core deliverable (correct,
  well-tested single-goal dynamic planning) solid; §4 documents the exact
  bottleneck-shortest-path alternative as the natural next extension for
  objective 4.
