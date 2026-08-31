"""
Scalability benchmark: random Cartesian graphs of increasing size, comparing
the cost of an INCREMENTAL replan (after a single transition availability
flip) against a COLD-START replan (rebuilding the planner from scratch),
to demonstrate the efficiency benefit of D* Lite's incremental updates.
"""

import random
import time
import statistics

from models import State, Transition, PlanningProblem
from planner import DStarLitePlanner, PlannerWeights


def random_problem(n_states=200, n_edges=800, n_bad=15, dim=3, seed=0):
    rng = random.Random(seed)
    states = [State(i, tuple(rng.uniform(0, 50) for _ in range(dim))) for i in range(n_states)]

    trans = []
    tid = 0
    # Ensure connectivity with a random spanning chain, then add random extra edges
    order = list(range(n_states))
    rng.shuffle(order)
    # bad states are sampled only from interior nodes, so start/goal are never bad
    bad_states = rng.sample(order[1:-1], min(n_bad, max(0, len(order) - 2)))
    for i in range(len(order) - 1):
        u, v = order[i], order[i + 1]
        trans.append(Transition(tid, u, v, cost=rng.uniform(0.5, 5.0),
                                 safety=rng.uniform(0.6, 1.0), reliability=rng.uniform(0.7, 1.0)))
        tid += 1
    while len(trans) < n_edges:
        u, v = rng.randrange(n_states), rng.randrange(n_states)
        if u == v:
            continue
        trans.append(Transition(tid, u, v, cost=rng.uniform(0.5, 5.0),
                                 safety=rng.uniform(0.6, 1.0), reliability=rng.uniform(0.7, 1.0)))
        tid += 1

    problem = PlanningProblem.build(states, trans, initial_state=order[0], goal_state=order[-1],
                                     bad_states=bad_states)
    return problem


def run_benchmark(sizes=((50, 300), (200, 800), (500, 2500), (1000, 6000))):
    print(f"{'N states':>10} {'N edges':>10} {'cold plan(ms)':>15} "
          f"{'incr replan(ms)':>17} {'speedup':>10} {'explored(cold)':>15} {'explored(incr)':>15}")
    for n_states, n_edges in sizes:
        problem = random_problem(n_states, n_edges, n_bad=max(5, n_states // 20), seed=42)
        planner = DStarLitePlanner(problem, PlannerWeights())
        result_cold = planner.plan()
        if not result_cold.success:
            print(f"{n_states:>10} {n_edges:>10}   (no path found for this random seed, skipping)")
            continue

        # Flip one transition on the current best path to force a local replan
        flip_id = result_cold.transition_path[len(result_cold.transition_path) // 2]

        t0 = time.perf_counter()
        result_incr = planner.set_transition_availability(flip_id, False)
        t_incr = (time.perf_counter() - t0) * 1000

        # Cold-start comparison: rebuild a fresh planner with the same modification
        problem2 = random_problem(n_states, n_edges, n_bad=max(5, n_states // 20), seed=42)
        problem2.transitions[flip_id].available = False
        planner_cold2 = DStarLitePlanner(problem2, PlannerWeights())
        t0 = time.perf_counter()
        result_cold2 = planner_cold2.plan()
        t_cold = (time.perf_counter() - t0) * 1000

        speedup = (t_cold / t_incr) if t_incr > 0 else float("inf")
        print(f"{n_states:>10} {n_edges:>10} {t_cold:>15.3f} {t_incr:>17.3f} "
              f"{speedup:>9.2f}x {result_cold2.states_explored:>15} {result_incr.states_explored:>15}")


if __name__ == "__main__":
    run_benchmark()
