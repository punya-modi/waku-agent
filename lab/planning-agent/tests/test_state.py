import operator

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from lab.planning_agent.state import (
    MAX_REPLANS, MAX_STEP_ATTEMPTS, PlanState, attempts, new_state, next_step,
)


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def test_plan_is_replaced_and_done_is_appended():
    """The two reducers are the whole point: plan overwrites, done accumulates."""
    graph = StateGraph(PlanState)

    def first(state):
        return {"plan": [{"id": "s1", "description": "one", "status": "pending"}],
                "done": [{"step_id": "s1", "digest": "d1", "artifacts": {}, "output_path": "p1"}]}

    def second(state):
        return {"plan": [{"id": "s2", "description": "two", "status": "pending"}],
                "done": [{"step_id": "s2", "digest": "d2", "artifacts": {}, "output_path": "p2"}]}

    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    app = graph.compile(checkpointer=InMemorySaver())

    out = app.invoke(new_state("g", budget_limit=1.0),
                     {"configurable": {"thread_id": "t"}})

    assert [s["id"] for s in out["plan"]] == ["s2"], "plan must be replaced"
    assert [r["step_id"] for r in out["done"]] == ["s1", "s2"], "done must append"


def test_next_step_skips_finished_steps_by_id():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "A", "status": "pending"},
                     {"id": "b", "description": "B", "status": "pending"}]
    state["done"] = [{"step_id": "a", "digest": "", "artifacts": {}, "output_path": ""}]
    assert next_step(state)["id"] == "b"


def test_next_step_survives_a_replan_that_shrinks_the_plan():
    """Review Focus 1. Two steps finished, then the replanner cut the plan to
    one. Selecting by len(done) would index past the end; selecting by id
    returns None, which is 'nothing left to do'."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "A", "status": "pending"}]
    state["done"] = [{"step_id": "a", "digest": "", "artifacts": {}, "output_path": ""},
                     {"step_id": "b", "digest": "", "artifacts": {}, "output_path": ""}]
    assert next_step(state) is None


def test_next_step_on_an_empty_plan_is_none():
    """Review Focus 2."""
    assert next_step(new_state("g", budget_limit=1.0)) is None


def test_new_state_starts_every_appended_list_empty():
    state = new_state("goal text", budget_limit=2.5)
    assert state["goal"] == "goal text"
    assert state["budget_limit"] == 2.5
    assert state["budget_spent"] == 0.0
    assert state["replans"] == 0
    assert "strategy" not in state, "the strategy split collapsed; see the preamble"
    for key in ("plan", "done", "revisions", "lessons", "rollups"):
        assert state[key] == [], key


def test_a_step_that_failed_twice_is_skipped_rather_than_retried_for_ever():
    """Review Focus 7. A step whose run_loop raises before its first model call
    costs nothing, so the budget never advances and the step is selected again
    until the recursion limit stops the task."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["lessons"] = [
        {"step_id": "a", "attempt": "x", "failure": "boom", "diagnosis": "d"},
        {"step_id": "a", "attempt": "x", "failure": "boom", "diagnosis": "d"},
    ]
    assert next_step(state)["id"] == "b", "'a' is exhausted and must be passed over"


def test_one_failure_is_not_enough_to_abandon_a_step():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["lessons"] = [{"step_id": "a", "attempt": "x", "failure": "boom",
                         "diagnosis": "d"}]
    assert next_step(state)["id"] == "a", "one retry is the point of a lesson"


def test_attempts_counts_lessons_for_one_step_only():
    state = new_state("g", budget_limit=1.0)
    state["lessons"] = [{"step_id": "a", "attempt": "", "failure": "", "diagnosis": ""},
                        {"step_id": "b", "attempt": "", "failure": "", "diagnosis": ""}]
    assert attempts(state, "a") == 1
    assert attempts(state, "zzz") == 0
