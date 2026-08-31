"""
Streamlit web app for the Safe Semantic Planner.

Run with:
    streamlit run app.py

Opens a local website (usually http://localhost:8501) where you can pick a
test case from the sidebar, see the graph rendered live, and interact with
the dynamic-environment scenarios (disable a transition, change the goal,
add a shortcut) to watch the planner replan in real time.
"""

import streamlit as st
import matplotlib.pyplot as plt

from planner import DStarLitePlanner, PlannerWeights
import scenarios
from visualize import draw_graph, legend

st.set_page_config(page_title="Safe Semantic Planner", layout="wide")

st.title("Safe Semantic Planner — D* Lite")
st.caption("PCCST503 Assignment 1 — interactive demo of all 6 illustrative test cases")

CASE_NAMES = {
    "1. Basic Reachability": 1,
    "2. Bad State Avoidance": 2,
    "3. Safety Margin Trade-off": 3,
    "4. Dynamic Transition": 4,
    "5. Goal Update": 5,
    "6. Transition Addition": 6,
    "Scalability Benchmark": 7,
}

choice = st.sidebar.radio("Test case", list(CASE_NAMES.keys()))
case_num = CASE_NAMES[choice]


def show_result(result, col=None):
    target = col if col is not None else st
    if not result.success:
        target.error("No valid path found.")
        return
    c1, c2, c3, c4 = target.columns(4)
    c1.metric("Total cost", f"{result.total_cost:.2f}")
    c2.metric("Min dist to bad state", f"{result.min_bad_distance:.2f}"
              if result.min_bad_distance != float("inf") else "n/a")
    c3.metric("States explored", result.states_explored)
    c4.metric("Planning time (ms)", f"{result.planning_time_s * 1000:.3f}")
    target.write(f"**Path:** {' → '.join(str(s) for s in result.state_path)}")


def render_figure(problem, labels, path_states, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_graph(ax, problem, labels, path_states, title)
    legend(fig)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------- #
if case_num == 1:
    st.header("Test Case 1: Basic Reachability")
    st.write("A single unique path S → A → B → G.")
    problem, weights, labels = scenarios.scenario_1()
    result = DStarLitePlanner(problem, weights).plan()
    render_figure(problem, labels, result.state_path, "S → A → B → G")
    show_result(result)

# --------------------------------------------------------------------- #
elif case_num == 2:
    st.header("Test Case 2: Bad State Avoidance")
    st.write("Two candidate routes exist; one passes through a bad state (X) "
             "and must never be chosen.")
    problem, weights, labels = scenarios.scenario_2()
    result = DStarLitePlanner(problem, weights).plan()
    render_figure(problem, labels, result.state_path, "Bad state avoided")
    show_result(result)
    if 2 in result.state_path:
        st.error("Bad state was visited — this should never happen!")
    else:
        st.success("Bad state X correctly avoided.")

# --------------------------------------------------------------------- #
elif case_num == 3:
    st.header("Test Case 3: Safety Margin Trade-off")
    st.write("Path 1 is cheaper but passes close to a bad state. Path 2 costs "
             "more but stays farther away. Adjust the weighting below to see "
             "the planner switch between them.")
    problem, w_cost, w_safety, labels = scenarios.scenario_3()

    mode = st.radio("Weighting", ["Cost-only", "Safety-weighted"], horizontal=True)
    if mode == "Cost-only":
        weights = w_cost
    else:
        weights = w_safety
        st.caption(f"w_unsafety={weights.w_unsafety}, w_bad_proximity={weights.w_bad_proximity}, "
                   f"safety_radius={weights.safety_radius}")

    with st.expander("Advanced: tune weights manually"):
        w_cost_v = st.slider("w_cost", 0.0, 5.0, weights.w_cost, 0.1)
        w_unsafety_v = st.slider("w_unsafety", 0.0, 5.0, weights.w_unsafety, 0.1)
        w_bad_v = st.slider("w_bad_proximity", 0.0, 10.0, weights.w_bad_proximity, 0.5)
        radius_v = st.slider("safety_radius", 0.0, 3.0, weights.safety_radius, 0.1)
        weights = PlannerWeights(w_cost=w_cost_v, w_unsafety=w_unsafety_v,
                                  w_unreliability=weights.w_unreliability,
                                  w_bad_proximity=w_bad_v, safety_radius=radius_v)

    result = DStarLitePlanner(problem, weights).plan()
    render_figure(problem, labels, result.state_path, f"{mode} weighting")
    show_result(result)

# --------------------------------------------------------------------- #
elif case_num == 4:
    st.header("Test Case 4: Dynamic Transition")
    st.write("Toggle the (A → G) shortcut off to force the planner to replan "
             "around it — live, without rebuilding the graph.")
    problem, weights, labels = scenarios.scenario_4()
    available = st.checkbox("Transition (A → G) available", value=True)

    planner = DStarLitePlanner(problem, weights)
    result_initial = planner.plan()
    if not available:
        result = planner.set_transition_availability(1, False)
    else:
        result = result_initial

    render_figure(problem, labels, result.state_path,
                  "(A,G) available" if available else "(A,G) removed — replanned")
    show_result(result)

# --------------------------------------------------------------------- #
elif case_num == 5:
    st.header("Test Case 5: Goal Update")
    st.write("Change the goal mid-execution and watch the planner produce a "
             "revised path.")
    problem, weights, labels = scenarios.scenario_5()
    goal_choice = st.selectbox("Goal state", options=list(problem.states.keys()),
                                index=list(problem.states.keys()).index(problem.goal_state),
                                format_func=lambda sid: f"{sid}: {labels.get(sid, sid)}")

    planner = DStarLitePlanner(problem, weights)
    planner.plan()
    result = planner.update_goal(goal_choice)
    render_figure(problem, labels, result.state_path, f"Goal = {goal_choice}")
    show_result(result)

# --------------------------------------------------------------------- #
elif case_num == 6:
    st.header("Test Case 6: Transition Addition")
    st.write("Insert a new shortcut transition and watch the planner "
             "discover the improved route.")
    problem, shortcut, weights, labels = scenarios.scenario_6()
    add_shortcut = st.checkbox("Add shortcut transition S → G (cost 1.2)", value=False)

    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    if add_shortcut:
        result = planner.add_transition(shortcut)
    else:
        result = result_before

    render_figure(problem, labels, result.state_path,
                  "Shortcut added" if add_shortcut else "No shortcut yet")
    show_result(result)
    if add_shortcut:
        st.success(f"Cost improved from {result_before.total_cost:.2f} to {result.total_cost:.2f}")

# --------------------------------------------------------------------- #
elif case_num == 7:
    st.header("Scalability Benchmark")
    st.write("Generates a random Cartesian graph and compares an incremental "
             "D* Lite replan (after disabling one transition) against a "
             "cold-start replan on a fresh planner.")
    from benchmark import random_problem

    n_states = st.slider("Number of states", 20, 1500, 300, 20)
    n_edges = st.slider("Number of edges", 50, 8000, max(150, n_states * 4), 50)
    n_bad = st.slider("Number of bad states", 0, max(1, n_states // 5), max(1, n_states // 20))

    if st.button("Run benchmark"):
        import time
        problem = random_problem(n_states, n_edges, n_bad=n_bad, seed=42)
        planner = DStarLitePlanner(problem, PlannerWeights())
        result_cold = planner.plan()
        if not result_cold.success:
            st.error("No path found for this random graph — try different sizes.")
        else:
            flip_id = result_cold.transition_path[len(result_cold.transition_path) // 2]

            t0 = time.perf_counter()
            result_incr = planner.set_transition_availability(flip_id, False)
            t_incr = (time.perf_counter() - t0) * 1000

            problem2 = random_problem(n_states, n_edges, n_bad=n_bad, seed=42)
            problem2.transitions[flip_id].available = False
            planner2 = DStarLitePlanner(problem2, PlannerWeights())
            t0 = time.perf_counter()
            result_cold2 = planner2.plan()
            t_cold = (time.perf_counter() - t0) * 1000

            c1, c2, c3 = st.columns(3)
            c1.metric("Cold-start replan (ms)", f"{t_cold:.2f}")
            c2.metric("Incremental replan (ms)", f"{t_incr:.2f}")
            c3.metric("Speedup", f"{(t_cold / t_incr if t_incr > 0 else float('inf')):.2f}x")
            st.bar_chart({"cold-start": t_cold, "incremental": t_incr})
            st.caption(f"States explored — cold: {result_cold2.states_explored}, "
                       f"incremental: {result_incr.states_explored}")

st.sidebar.divider()
st.sidebar.caption("Safe Semantic Planner — D* Lite implementation")
