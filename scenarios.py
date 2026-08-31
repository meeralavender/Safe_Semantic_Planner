"""
Shared builders for the six illustrative test cases from the assignment
PDF. Kept in one place so demo.py (correctness checks) and visualize.py
(plots) always describe the exact same graphs.
"""

from models import State, Transition, PlanningProblem
from planner import PlannerWeights


def scenario_1():
    """Basic Reachability: S -> A -> B -> G"""
    states = [State(0, (0, 0)), State(1, (1, 0)), State(2, (2, 0)), State(3, (3, 0))]
    trans = [
        Transition(0, 0, 1, cost=1.0, safety=1.0, reliability=1.0),
        Transition(1, 1, 2, cost=1.0, safety=1.0, reliability=1.0),
        Transition(2, 2, 3, cost=1.0, safety=1.0, reliability=1.0),
    ]
    problem = PlanningProblem.build(states, trans, initial_state=0, goal_state=3)
    labels = {0: "S", 1: "A", 2: "B", 3: "G"}
    return problem, PlannerWeights(), labels


def scenario_2():
    """Bad State Avoidance: S->A->X->G (X bad) vs S->C->D->G"""
    states = [
        State(0, (0, 0)), State(1, (1, 1)), State(2, (2, 1)), State(3, (4, 0)),
        State(4, (1, -1)), State(5, (2, -1)),
    ]
    trans = [
        Transition(0, 0, 1, cost=1.0, safety=1.0, reliability=1.0),
        Transition(1, 1, 2, cost=1.0, safety=0.9, reliability=1.0),
        Transition(2, 2, 3, cost=1.0, safety=0.9, reliability=1.0),
        Transition(3, 0, 4, cost=1.5, safety=1.0, reliability=1.0),
        Transition(4, 4, 5, cost=1.5, safety=1.0, reliability=1.0),
        Transition(5, 5, 3, cost=1.5, safety=1.0, reliability=1.0),
    ]
    problem = PlanningProblem.build(states, trans, initial_state=0, goal_state=3, bad_states=[2])
    labels = {0: "S", 1: "A", 2: "X(bad)", 3: "G", 4: "C", 5: "D"}
    return problem, PlannerWeights(), labels


def scenario_3():
    """Safety Margin Trade-off. Returns (problem, weights_cost_only, weights_safety, labels)."""
    states = [
        State(0, (0, 0)),
        State(1, (1, 0.2)), State(2, (2, 0.2)),
        State(3, (1, -2)), State(4, (2, -2)),
        State(5, (3, 0)),
        State(6, (2, 1)),  # bad state
    ]
    trans = [
        Transition(0, 0, 1, cost=1.0, safety=0.95, reliability=1.0),
        Transition(1, 1, 2, cost=1.0, safety=0.95, reliability=1.0),
        Transition(2, 2, 5, cost=1.0, safety=0.95, reliability=1.0),
        Transition(3, 0, 3, cost=2.0, safety=0.99, reliability=1.0),
        Transition(4, 3, 4, cost=2.0, safety=0.99, reliability=1.0),
        Transition(5, 4, 5, cost=2.0, safety=0.99, reliability=1.0),
    ]
    problem = PlanningProblem.build(states, trans, initial_state=0, goal_state=5, bad_states=[6])
    weights_cost_only = PlannerWeights(w_cost=1.0, w_unsafety=0.0, w_unreliability=0.0, w_bad_proximity=0.0)
    weights_safety = PlannerWeights(w_cost=1.0, w_unsafety=1.0, w_unreliability=0.0,
                                     w_bad_proximity=5.0, safety_radius=1.5)
    labels = {0: "S", 1: "P1a", 2: "P1b", 3: "P2a", 4: "P2b", 5: "G", 6: "bad"}
    return problem, weights_cost_only, weights_safety, labels


def scenario_4():
    """Dynamic Transition: (A,G) becomes unavailable."""
    states = [State(0, (0, 0)), State(1, (1, 0)), State(2, (2, 0)),
              State(3, (1, -1)), State(4, (2, -1))]
    trans = [
        Transition(0, 0, 1, cost=1.0, safety=1.0, reliability=1.0),
        Transition(1, 1, 2, cost=1.0, safety=1.0, reliability=1.0),
        Transition(2, 0, 3, cost=1.5, safety=1.0, reliability=1.0),
        Transition(3, 3, 4, cost=1.5, safety=1.0, reliability=1.0),
        Transition(4, 4, 2, cost=1.5, safety=1.0, reliability=1.0),
    ]
    problem = PlanningProblem.build(states, trans, initial_state=0, goal_state=2)
    labels = {0: "S", 1: "A", 2: "G", 3: "C", 4: "D"}
    return problem, PlannerWeights(), labels


def scenario_5():
    """Goal Update: goal changes from 3 to 2 mid-execution."""
    states = [State(0, (0, 0)), State(1, (1, 0)), State(2, (2, 0)), State(3, (3, 0))]
    trans = [
        Transition(0, 0, 1, cost=1.0, safety=1.0, reliability=1.0),
        Transition(1, 1, 2, cost=1.0, safety=1.0, reliability=1.0),
        Transition(2, 2, 3, cost=1.0, safety=1.0, reliability=1.0),
    ]
    problem = PlanningProblem.build(states, trans, initial_state=0, goal_state=3)
    labels = {0: "S", 1: "A", 2: "G2(new)", 3: "G1(old)"}
    return problem, PlannerWeights(), labels


def scenario_6():
    """Transition Addition: a shortcut S->G is inserted."""
    states = [State(0, (0, 0)), State(1, (1, 0)), State(2, (2, 0)), State(3, (3, 0))]
    trans = [
        Transition(0, 0, 1, cost=1.0, safety=1.0, reliability=1.0),
        Transition(1, 1, 2, cost=1.0, safety=1.0, reliability=1.0),
        Transition(2, 2, 3, cost=1.0, safety=1.0, reliability=1.0),
    ]
    problem = PlanningProblem.build(states, trans, initial_state=0, goal_state=3)
    shortcut = Transition(99, 0, 3, cost=1.2, safety=0.98, reliability=1.0)
    labels = {0: "S", 1: "A", 2: "B", 3: "G"}
    return problem, shortcut, PlannerWeights(), labels
