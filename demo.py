"""
Runs the six illustrative test cases from the assignment PDF against the
DStarLitePlanner, printing the metrics the assignment asks students to
evaluate: goal success rate, bad states visited, total path cost, minimum
distance to bad states, states explored, planning time, memory usage and
replanning time.
"""

from planner import DStarLitePlanner
import scenarios


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def report(label, result):
    print(f"  {label}")
    print(f"    success            : {result.success}")
    if result.success:
        print(f"    state path         : {result.state_path}")
        print(f"    transition path    : {result.transition_path}")
        print(f"    total cost         : {result.total_cost:.3f}")
        print(f"    avg safety score   : {result.safety_score:.3f}")
        print(f"    min dist to bad    : {result.min_bad_distance:.3f}")
    print(f"    states explored    : {result.states_explored}")
    print(f"    planning time (ms) : {result.planning_time_s * 1000:.4f}")
    print(f"    peak memory (KB)   : {result.peak_memory_kb:.2f}")


def test_case_1():
    line("Test Case 1: Basic Reachability (S -> A -> B -> G)")
    problem, weights, _ = scenarios.scenario_1()
    planner = DStarLitePlanner(problem, weights)
    result = planner.plan()
    report("Unique valid path expected: [0,1,2,3]", result)
    assert result.success and result.state_path == [0, 1, 2, 3]
    return result


def test_case_2():
    line("Test Case 2: Bad State Avoidance")
    problem, weights, _ = scenarios.scenario_2()
    planner = DStarLitePlanner(problem, weights)
    result = planner.plan()
    report("Second path (S->C->D->G) must be selected", result)
    assert result.success and 2 not in result.state_path
    assert result.state_path == [0, 4, 5, 3]
    return result


def test_case_3():
    line("Test Case 3: Safety Margin Trade-off")
    problem, weights_cost_only, weights_safety, _ = scenarios.scenario_3()

    planner_cost_only = DStarLitePlanner(problem, weights_cost_only)
    result_cost_only = planner_cost_only.plan()
    report("Cost-only weighting -> picks the cheap-but-close path", result_cost_only)

    planner_safety = DStarLitePlanner(problem, weights_safety)
    result_safety = planner_safety.plan()
    report("Safety-weighted -> trades cost for distance from bad state", result_safety)
    return result_cost_only, result_safety


def test_case_4():
    line("Test Case 4: Dynamic Transition (edge removed)")
    problem, weights, _ = scenarios.scenario_4()
    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    report("Before: S->A->G", result_before)
    assert result_before.state_path == [0, 1, 2]

    result_after = planner.set_transition_availability(1, False)
    report("After (A,G) unavailable: alternative path found", result_after)
    assert result_after.success and result_after.state_path == [0, 3, 4, 2]
    return result_before, result_after


def test_case_5():
    line("Test Case 5: Goal Update")
    problem, weights, _ = scenarios.scenario_5()
    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    report("Original goal (3)", result_before)

    result_after = planner.update_goal(2)
    report("Revised goal (2), replanned without rebuilding the graph", result_after)
    assert result_after.success and result_after.state_path == [0, 1, 2]
    return result_before, result_after


def test_case_6():
    line("Test Case 6: Transition Addition (shortcut)")
    problem, shortcut, weights, _ = scenarios.scenario_6()
    planner = DStarLitePlanner(problem, weights)
    result_before = planner.plan()
    report("Before shortcut: S->A->B->G, cost 3", result_before)

    result_after = planner.add_transition(shortcut)
    report("After shortcut S->G added: improved solution found", result_after)
    assert result_after.success and result_after.total_cost < result_before.total_cost
    assert result_after.state_path == [0, 3]
    return result_before, result_after


if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    line("ALL TEST CASES PASSED")
