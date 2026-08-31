"""
Visualizes each test-case graph with matplotlib: states as points, bad
states in red, transitions as arrows, and the planner's chosen path
highlighted in bold.

Run directly to open a matplotlib window for each test case one at a time
(close a window to see the next one):

    python3 visualize.py

Or save every figure to PNG files instead of opening windows (useful when
there's no display, e.g. over SSH, or to drop images straight into your
report):

    python3 visualize.py --save
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from planner import DStarLitePlanner
import scenarios

OUT_DIR = os.path.join(os.path.dirname(__file__), "visuals")


def draw_graph(ax, problem, labels, path_states=None, title=""):
    pos = {sid: (s.embedding[0], s.embedding[1] if len(s.embedding) > 1 else 0)
           for sid, s in problem.states.items()}
    path_edges = set()
    if path_states:
        path_edges = set(zip(path_states, path_states[1:]))

    # draw all transitions
    for t in problem.transitions.values():
        x1, y1 = pos[t.src]
        x2, y2 = pos[t.dst]
        on_path = (t.src, t.dst) in path_edges
        color = "#1f77b4" if on_path else ("#cccccc" if t.available else "#ffbdbd")
        width = 3.0 if on_path else 1.2
        style = "-" if t.available else "--"
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=width, linestyle=style,
                                 shrinkA=14, shrinkB=14, zorder=2 if on_path else 1)
        ax.add_patch(arrow)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate(f"c={t.cost:g}", (mx, my), fontsize=7, color="#555555",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # draw states
    for sid, (x, y) in pos.items():
        if sid in problem.bad_states:
            face, edge, marker, size = "#e74c3c", "#a93226", "X", 220
        elif sid == problem.initial_state:
            face, edge, marker, size = "#2ecc71", "#1e8449", "o", 220
        elif sid == problem.goal_state:
            face, edge, marker, size = "#f1c40f", "#b7950b", "*", 320
        else:
            face, edge, marker, size = "#aed6f1", "#2874a6", "o", 180
        ax.scatter([x], [y], s=size, c=face, edgecolors=edge, marker=marker,
                   linewidths=1.5, zorder=3)
        label = labels.get(sid, str(sid))
        ax.annotate(f"{sid}:{label}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.25)
    ax.axis("off")


def legend(fig):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#f1c40f', markersize=14, label='Goal'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c', markersize=10, label='Bad state'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#aed6f1', markersize=10, label='State'),
        Line2D([0], [0], color='#1f77b4', linewidth=3, label='Chosen path'),
        Line2D([0], [0], color='#cccccc', linewidth=1.2, label='Available transition'),
        Line2D([0], [0], color='#ffbdbd', linewidth=1.2, linestyle='--', label='Unavailable transition'),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)


def show_or_save(fig, name, save):
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_case_1(save):
    problem, weights, labels = scenarios.scenario_1()
    result = DStarLitePlanner(problem, weights).plan()
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_graph(ax, problem, labels, result.state_path,
               f"Test Case 1: Basic Reachability  (cost={result.total_cost:.2f})")
    legend(fig)
    show_or_save(fig, "case1_basic_reachability", save)


def visualize_case_2(save):
    problem, weights, labels = scenarios.scenario_2()
    result = DStarLitePlanner(problem, weights).plan()
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_graph(ax, problem, labels, result.state_path,
               f"Test Case 2: Bad State Avoidance  (cost={result.total_cost:.2f})")
    legend(fig)
    show_or_save(fig, "case2_bad_state_avoidance", save)


def visualize_case_3(save):
    problem, w_cost, w_safety, labels = scenarios.scenario_3()
    result_cost = DStarLitePlanner(problem, w_cost).plan()
    result_safe = DStarLitePlanner(problem, w_safety).plan()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_graph(axes[0], problem, labels, result_cost.state_path,
               f"Cost-only  (cost={result_cost.total_cost:.2f}, min-dist-to-bad={result_cost.min_bad_distance:.2f})")
    draw_graph(axes[1], problem, labels, result_safe.state_path,
               f"Safety-weighted  (cost={result_safe.total_cost:.2f}, min-dist-to-bad={result_safe.min_bad_distance:.2f})")
    fig.suptitle("Test Case 3: Safety Margin Trade-off", fontsize=12)
    legend(fig)
    show_or_save(fig, "case3_safety_margin_tradeoff", save)


def visualize_case_4(save):
    problem, weights, labels = scenarios.scenario_4()
    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    result_after = planner.set_transition_availability(1, False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_graph(axes[0], problem, labels, result_before.state_path,
               f"Before: (A,G) available  (cost={result_before.total_cost:.2f})")
    draw_graph(axes[1], problem, labels, result_after.state_path,
               f"After: (A,G) removed  (cost={result_after.total_cost:.2f})")
    fig.suptitle("Test Case 4: Dynamic Transition", fontsize=12)
    legend(fig)
    show_or_save(fig, "case4_dynamic_transition", save)


def visualize_case_5(save):
    problem, weights, labels = scenarios.scenario_5()
    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    result_after = planner.update_goal(2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    draw_graph(axes[0], problem, labels, result_before.state_path,
               f"Original goal=3  (cost={result_before.total_cost:.2f})")
    draw_graph(axes[1], problem, labels, result_after.state_path,
               f"Updated goal=2  (cost={result_after.total_cost:.2f})")
    fig.suptitle("Test Case 5: Goal Update", fontsize=12)
    legend(fig)
    show_or_save(fig, "case5_goal_update", save)


def visualize_case_6(save):
    problem, shortcut, weights, labels = scenarios.scenario_6()
    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    result_after = planner.add_transition(shortcut)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    draw_graph(axes[0], problem, labels, result_before.state_path,
               f"Before shortcut  (cost={result_before.total_cost:.2f})")
    draw_graph(axes[1], problem, labels, result_after.state_path,
               f"After shortcut S->G added  (cost={result_after.total_cost:.2f})")
    fig.suptitle("Test Case 6: Transition Addition", fontsize=12)
    legend(fig)
    show_or_save(fig, "case6_transition_addition", save)


ALL_CASES = [visualize_case_1, visualize_case_2, visualize_case_3,
             visualize_case_4, visualize_case_5, visualize_case_6]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                         help="Save PNGs to visuals/ instead of opening windows")
    parser.add_argument("--case", type=int, choices=range(1, 7),
                         help="Only visualize a single test case (1-6)")
    args = parser.parse_args()

    cases = [ALL_CASES[args.case - 1]] if args.case else ALL_CASES
    for case_fn in cases:
        case_fn(args.save)
