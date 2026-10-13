import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from lab.planning_agent.graph import after_plan, after_step, build_graph
from lab.planning_agent.state import new_state

CONFIG = {"configurable": {"thread_id": "t"}}


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def _result(step_id):
    return {"step_id": step_id, "digest": step_id, "artifacts": {}, "output_path": ""}


def _plan_of(*ids):
    def plan_fn(state):
        return {"plan": _steps(*ids)}
    return plan_fn


def _executor(log):
    def execute_fn(state):
        from lab.planning_agent.state import next_step
        step = next_step(state)
        log.append(step["id"])
        return {"done": [_result(step["id"])], "budget_spent": state["budget_spent"] + 0.1}
    return execute_fn


def _noop(state):
    return {}


def _synth(state):
    return {"last_error": ""}


def test_an_empty_plan_routes_straight_to_synthesize():
    """Review Focus 2."""
    state = new_state("g", budget_limit=5.0)
    assert after_plan(state) == "synthesize"


def test_a_zero_budget_halts_before_any_step_runs():
    """Review Focus 5. The check sits after `plan`, so nothing is paid for."""
    state = new_state("g", budget_limit=0.0)
    state["plan"] = _steps("a")
    assert after_plan(state) == "synthesize"


def test_a_negative_budget_also_halts():
    state = new_state("g", budget_limit=-1.0)
    state["plan"] = _steps("a")
    assert after_plan(state) == "synthesize"


def test_a_plan_with_work_and_budget_routes_to_the_first_step():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a")
    assert after_plan(state) == "step"


def test_after_a_step_more_work_routes_back_to_the_step_node():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["done"] = [_result("a")]
    assert after_step(state) == "step"


def test_after_the_last_step_routes_to_synthesize():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a")
    state["done"] = [_result("a")]
    assert after_step(state) == "synthesize"


def test_a_requested_replan_wins_over_remaining_work():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["done"] = [_result("a")]
    state["replan_requested"] = True
    assert after_step(state) == "replan"


def test_an_exhausted_budget_stops_the_cycle_even_with_work_left():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["done"] = [_result("a")]
    state["budget_spent"] = 1.0
    assert after_step(state) == "synthesize"


def test_a_replan_is_refused_once_the_cap_is_reached():
    """Review Focus 7. Without this the cycle runs until the recursion limit,
    and every lap costs a model call once the nodes are real."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["replan_requested"] = True
    state["replans"] = 5
    assert after_step(state) == "step", "work is left, so carry on without replanning"

    state["done"] = [_result("a"), _result("b")]
    assert after_step(state) == "synthesize", "and answer when nothing is left"


def test_a_replan_under_the_cap_still_wins():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["replan_requested"] = True
    state["replans"] = 4
    assert after_step(state) == "replan"


def test_the_replan_edge_does_not_buy_a_step_past_the_budget():
    """Review Focus 6. The replan node runs, and then the same budget gate
    that guards the first step guards the next one."""
    log: list[str] = []

    def failing_once(state):
        # One step that asks for a replan without spending anything, which is
        # what a provider refusal looks like.
        return {"replan_requested": True, "budget_spent": state["budget_limit"]}

    app = build_graph(plan_fn=_plan_of("a", "b"), execute_fn=failing_once,
                      replan_fn=lambda s: {"replan_requested": False,
                                           "replans": s["replans"] + 1},
                      synthesize_fn=_synth, checkpointer=InMemorySaver())
    out = app.invoke(new_state("g", budget_limit=1.0), CONFIG)
    assert out["done"] == [], "no step completed"
    assert out["replans"] == 1, "one replan ran, and then the budget stopped it"


def test_the_cycle_runs_every_step_once_and_stops():
    log: list[str] = []
    app = build_graph(plan_fn=_plan_of("a", "b", "c"), execute_fn=_executor(log),
                      replan_fn=_noop, synthesize_fn=_synth,
                      checkpointer=InMemorySaver())
    out = app.invoke(new_state("g", budget_limit=10.0), CONFIG)
    assert log == ["a", "b", "c"]
    assert [r["step_id"] for r in out["done"]] == ["a", "b", "c"]


def test_the_cycle_stops_when_the_budget_runs_out_midway():
    log: list[str] = []
    app = build_graph(plan_fn=_plan_of("a", "b", "c"), execute_fn=_executor(log),
                      replan_fn=_noop, synthesize_fn=_synth,
                      checkpointer=InMemorySaver())
    app.invoke(new_state("g", budget_limit=0.2), CONFIG)
    assert log == ["a", "b"], (
        "0.1 a step against a 0.2 limit stops after two: the guard is "
        "spent >= limit, so 0.2 >= 0.2 halts where 0.2 >= 0.25 would not")


def test_an_empty_plan_reaches_the_end_without_executing_anything():
    log: list[str] = []
    app = build_graph(plan_fn=lambda s: {"plan": []}, execute_fn=_executor(log),
                      replan_fn=_noop, synthesize_fn=_synth,
                      checkpointer=InMemorySaver())
    app.invoke(new_state("g", budget_limit=10.0), CONFIG)
    assert log == []
