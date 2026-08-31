"""
Safe Semantic Planner — D* Lite implementation.

Implements the incremental D* Lite algorithm (Koenig & Likhachev, 2005),
adapted so that:
  * bad states are hard-excluded from the search graph (never expanded,
    never used as a successor/predecessor),
  * each directed transition's search weight blends its raw cost with a
    safety/reliability penalty and a proximity-to-bad-state penalty, so the
    planner naturally prefers paths that stay far from bad states,
  * the graph can be updated incrementally (edge cost/availability change,
    edge add/remove, bad-state set change, goal change) and replanned
    without discarding all previously computed g/rhs values.

Time complexity (per replan after a *local* change such as a single edge
cost/availability update): O((k + E_affected) log V) where k is the number
of vertices whose key changes propagate to, using a binary-heap priority
queue -- identical asymptotic behaviour to the original D* Lite. A full
cold-start plan is O(E log V), same as Dijkstra/A*.

Space complexity: O(V + E) for the graph, g/rhs tables and the open list.
"""

from __future__ import annotations
import heapq
import itertools
import math
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

from models import PlanningProblem, PlanningResult, State, Transition

INF = math.inf


@dataclass
class PlannerWeights:
    """Weights that turn the multi-objective Score(P) = aG - bC + gD + dR
    (from the assignment) into a single additive edge weight D* Lite can
    optimise. See DESIGN_REPORT.md, section 4, for the justification."""
    w_cost: float = 1.0          # beta  : weight on raw transition cost
    w_unsafety: float = 2.0      # penalises low per-edge 'safety' field
    w_unreliability: float = 1.0 # penalises low per-edge 'reliability' field
    w_bad_proximity: float = 3.0 # penalises destination states close to bad states
    safety_radius: float = 2.0   # distance (in embedding space) considered "safe enough"


class DStarLitePlanner:
    def __init__(self, problem: PlanningProblem, weights: PlannerWeights = None):
        self.weights = weights or PlannerWeights()
        self._load(problem)

    # ------------------------------------------------------------------ #
    # Graph construction / maintenance
    # ------------------------------------------------------------------ #
    def _load(self, problem: PlanningProblem):
        self.states: Dict[int, State] = dict(problem.states)
        self.bad_states: Set[int] = set(problem.bad_states)
        self.start = problem.initial_state
        self.goal = problem.goal_state

        self.out_edges: Dict[int, List[Transition]] = {sid: [] for sid in self.states}
        self.in_edges: Dict[int, List[Transition]] = {sid: [] for sid in self.states}
        self.transitions: Dict[int, Transition] = {}
        for t in problem.transitions.values():
            self._index_transition(t)

        self._precompute_bad_distances()
        self._init_search()

    def _index_transition(self, t: Transition):
        self.transitions[t.id] = t
        self.out_edges.setdefault(t.src, []).append(t)
        self.in_edges.setdefault(t.dst, []).append(t)

    def _precompute_bad_distances(self):
        """Euclidean distance from every state to the nearest bad state."""
        self._bad_dist: Dict[int, float] = {}
        for sid, s in self.states.items():
            if not self.bad_states:
                self._bad_dist[sid] = INF
                continue
            self._bad_dist[sid] = min(
                s.distance_to(self.states[b]) for b in self.bad_states if b in self.states
            )

    # ------------------------------------------------------------------ #
    # Edge weight: blends cost, safety, reliability, bad-state proximity
    # ------------------------------------------------------------------ #
    def edge_weight(self, t: Transition) -> float:
        if not t.available or t.src in self.bad_states or t.dst in self.bad_states:
            return INF
        w = self.weights
        unsafety = max(0.0, 1.0 - t.safety)
        unreliability = max(0.0, 1.0 - t.reliability)
        dist = self._bad_dist.get(t.dst, INF)
        proximity_penalty = 0.0
        if dist < w.safety_radius:
            proximity_penalty = (w.safety_radius - dist)
        weight = (w.w_cost * t.cost
                  + w.w_unsafety * unsafety
                  + w.w_unreliability * unreliability
                  + w.w_bad_proximity * proximity_penalty)
        return weight

    def successors(self, sid: int) -> List[Transition]:
        if sid in self.bad_states:
            return []
        return [t for t in self.out_edges.get(sid, []) if t.available and t.dst not in self.bad_states]

    def predecessors(self, sid: int) -> List[Transition]:
        if sid in self.bad_states:
            return []
        return [t for t in self.in_edges.get(sid, []) if t.available and t.src not in self.bad_states]

    # ------------------------------------------------------------------ #
    # D* Lite core
    # ------------------------------------------------------------------ #
    def _init_search(self):
        self.g: Dict[int, float] = {sid: INF for sid in self.states}
        self.rhs: Dict[int, float] = {sid: INF for sid in self.states}
        self.km = 0.0
        self._counter = itertools.count()
        self.open_heap: List[Tuple[Tuple[float, float], int, int]] = []
        self.open_set: Dict[int, Tuple[float, float]] = {}
        self.states_explored = 0

        if self.goal in self.states:
            self.rhs[self.goal] = 0.0
            self._push(self.goal)

    def _heuristic(self, sid: int) -> float:
        """Admissible heuristic: scaled Euclidean distance from start to sid.
        Scale = the smallest (weight / distance) ratio observed across all
        edges, which guarantees h never overestimates true edge weight per
        unit distance -> consistency is preserved."""
        if not hasattr(self, "_h_scale"):
            self._h_scale = self._compute_heuristic_scale()
        if self.start not in self.states or sid not in self.states:
            return 0.0
        d = self.states[self.start].distance_to(self.states[sid])
        return self._h_scale * d

    def _compute_heuristic_scale(self) -> float:
        ratios = []
        for t in self.transitions.values():
            if not t.available or t.src == t.dst:
                continue
            if t.src not in self.states or t.dst not in self.states:
                continue
            d = self.states[t.src].distance_to(self.states[t.dst])
            w = self.edge_weight(t)
            if d > 1e-9 and w < INF:
                ratios.append(w / d)
        return min(ratios) if ratios else 0.0

    def _key(self, sid: int) -> Tuple[float, float]:
        g_rhs = min(self.g[sid], self.rhs[sid])
        return (g_rhs + self._heuristic(sid) + self.km, g_rhs)

    def _push(self, sid: int):
        key = self._key(sid)
        self.open_set[sid] = key
        heapq.heappush(self.open_heap, (key, next(self._counter), sid))

    def _pop_valid_top(self):
        """Lazily pops stale heap entries until a valid top is found."""
        while self.open_heap:
            key, _, sid = self.open_heap[0]
            if sid in self.open_set and self.open_set[sid] == key:
                return key, sid
            heapq.heappop(self.open_heap)
        return None

    def _top_key(self) -> Tuple[float, float]:
        top = self._pop_valid_top()
        return top[0] if top else (INF, INF)

    def update_vertex(self, sid: int):
        if sid != self.goal:
            best = INF
            for t in self.successors(sid):
                w = self.edge_weight(t)
                if w < INF:
                    best = min(best, w + self.g.get(t.dst, INF))
            self.rhs[sid] = best
        if sid in self.open_set:
            del self.open_set[sid]  # lazy delete; stale heap entries ignored later
        if self.g[sid] != self.rhs[sid]:
            self._push(sid)

    def compute_shortest_path(self):
        while True:
            top = self._pop_valid_top()
            top_key = top[0] if top else (INF, INF)
            start_key = self._key(self.start)
            if not (top_key < start_key) and self.rhs[self.start] == self.g[self.start]:
                break
            if top is None:
                break
            _, u = top
            heapq.heappop(self.open_heap)
            del self.open_set[u]
            self.states_explored += 1
            u_key_new = self._key(u)
            if top_key < u_key_new:
                self._push(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for t in self.predecessors(u):
                    self.update_vertex(t.src)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for t in self.predecessors(u):
                    self.update_vertex(t.src)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan(self) -> PlanningResult:
        tracemalloc.start()
        t0 = time.perf_counter()
        self.compute_shortest_path()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return self._extract_result(elapsed, peak / 1024.0)

    def _extract_result(self, elapsed: float, peak_kb: float) -> PlanningResult:
        if self.g.get(self.start, INF) == INF or self.start in self.bad_states:
            return PlanningResult(success=False, states_explored=self.states_explored,
                                   planning_time_s=elapsed, peak_memory_kb=peak_kb)

        path_states = [self.start]
        path_transitions = []
        total_cost = 0.0
        min_bad_dist = self._bad_dist.get(self.start, INF)
        cur = self.start
        visited = {cur}
        while cur != self.goal:
            best_t, best_val = None, INF
            for t in self.successors(cur):
                w = self.edge_weight(t)
                if w >= INF:
                    continue
                val = w + self.g.get(t.dst, INF)
                if val < best_val:
                    best_val, best_t = val, t
            if best_t is None:
                return PlanningResult(success=False, states_explored=self.states_explored,
                                       planning_time_s=elapsed, peak_memory_kb=peak_kb)
            total_cost += best_t.cost
            path_transitions.append(best_t.id)
            cur = best_t.dst
            min_bad_dist = min(min_bad_dist, self._bad_dist.get(cur, INF))
            path_states.append(cur)
            if cur in visited:  # safety guard against cycles
                return PlanningResult(success=False, states_explored=self.states_explored,
                                       planning_time_s=elapsed, peak_memory_kb=peak_kb)
            visited.add(cur)

        avg_safety = (sum(self.transitions[tid].safety for tid in path_transitions) / len(path_transitions)
                      if path_transitions else 1.0)
        return PlanningResult(
            success=True,
            state_path=path_states,
            transition_path=path_transitions,
            total_cost=total_cost,
            safety_score=avg_safety,
            min_bad_distance=min_bad_dist,
            states_explored=self.states_explored,
            planning_time_s=elapsed,
            peak_memory_kb=peak_kb,
        )

    # ------------------------------------------------------------------ #
    # Dynamic environment updates (incremental replanning)
    # ------------------------------------------------------------------ #
    def _reset_heuristic_cache(self):
        if hasattr(self, "_h_scale"):
            del self._h_scale

    def set_transition_availability(self, transition_id: int, available: bool) -> PlanningResult:
        t = self.transitions[transition_id]
        t.available = available
        self.update_vertex(t.src)
        for pt in self.predecessors(t.src):
            self.update_vertex(pt.src)
        return self.plan()

    def set_transition_cost(self, transition_id: int, cost: float) -> PlanningResult:
        t = self.transitions[transition_id]
        t.cost = cost
        self.update_vertex(t.src)
        return self.plan()

    def add_transition(self, t: Transition) -> PlanningResult:
        self._index_transition(t)
        self._reset_heuristic_cache()
        self.update_vertex(t.src)
        return self.plan()

    def remove_transition(self, transition_id: int) -> PlanningResult:
        return self.set_transition_availability(transition_id, False)

    def update_bad_states(self, bad_states: List[int]) -> PlanningResult:
        self.bad_states = set(bad_states)
        self._precompute_bad_distances()
        self._reset_heuristic_cache()
        for sid in self.states:
            self.update_vertex(sid)
        return self.plan()

    def update_goal(self, new_goal: int) -> PlanningResult:
        """Goal changes invalidate the backward search direction D* Lite
        relies on, so (as documented in the assignment / D* Lite literature)
        this is the one update that requires re-initialising rhs/g -- it
        cannot be done as cheaply as an edge-cost update. We still avoid
        rebuilding the *graph* (adjacency, bad-state distances, heuristic
        scale), only the search state is reset."""
        self.goal = new_goal
        self.km = 0.0
        self.g = {sid: INF for sid in self.states}
        self.rhs = {sid: INF for sid in self.states}
        self.open_heap.clear()
        self.open_set.clear()
        if self.goal in self.states:
            self.rhs[self.goal] = 0.0
            self._push(self.goal)
        return self.plan()

    def update_start(self, new_start: int):
        """Cheap start move, as in classic D* Lite (km += h(old_start))."""
        self.km += self._heuristic(self.start)
        self.start = new_start
