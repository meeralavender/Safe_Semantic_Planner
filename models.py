"""
Core data model for the Safe Semantic Planner.

Mirrors the C++ interfaces suggested in the assignment PDF
(State, Transition, PlanningProblem, PlanningResult) as Python dataclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math


@dataclass
class State:
    """A state embedded in a finite Cartesian space R^d."""
    id: int
    embedding: Tuple[float, ...]

    def distance_to(self, other: "State") -> float:
        return math.dist(self.embedding, other.embedding)


@dataclass
class Transition:
    """A directed transition between two states with cost/quality attributes."""
    id: int
    src: int
    dst: int
    cost: float
    safety: float          # in [0, 1], higher = safer
    reliability: float     # in [0, 1], higher = more reliable
    available: bool = True


@dataclass
class PlanningProblem:
    initial_state: int
    goal_state: int
    bad_states: List[int]
    states: Dict[int, State]
    transitions: Dict[int, Transition]

    @staticmethod
    def build(states: List[State], transitions: List[Transition],
              initial_state: int, goal_state: int,
              bad_states: List[int] = None) -> "PlanningProblem":
        return PlanningProblem(
            initial_state=initial_state,
            goal_state=goal_state,
            bad_states=list(bad_states or []),
            states={s.id: s for s in states},
            transitions={t.id: t for t in transitions},
        )


@dataclass
class PlanningResult:
    success: bool
    state_path: List[int] = field(default_factory=list)
    transition_path: List[int] = field(default_factory=list)
    total_cost: float = math.inf
    safety_score: float = 0.0
    min_bad_distance: float = math.inf
    states_explored: int = 0
    planning_time_s: float = 0.0
    peak_memory_kb: float = 0.0

    def __repr__(self):
        if not self.success:
            return f"PlanningResult(FAILED, explored={self.states_explored})"
        return (f"PlanningResult(path={self.state_path}, cost={self.total_cost:.3f}, "
                f"min_bad_dist={self.min_bad_distance:.3f}, explored={self.states_explored}, "
                f"time={self.planning_time_s*1000:.3f}ms)")
