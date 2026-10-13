import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.replan import make_replanner
from lab.planning_agent.state import new_state


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def _result(step_id):
    return {"step_id": step_id, "digest": step_id, "artifacts": {}, "output_path": ""}


def _state_with_a_failure():
    state = new_state("g", budget_limit=10.0)
    state["plan"] = _steps("a", "b", "c")
    state["done"] = [_result("a")]
    state["lessons"] = [{"step_id": "b", "attempt": "called the venue",
                         "failure": "line was dead", "diagnosis": "use email"}]
    state["replan_requested"] = True
    state["last_error"] = "RuntimeError: line was dead"
    return state


def test_replanning_keeps_finished_work_and_restructures_the_rest():
    merged = make_replanner(lambda **kw: _steps("b2", "c"))(_state_with_a_failure())
    assert [s["id"] for s in merged["plan"]] == ["b2", "c"]
    assert "done" not in merged, "finished work is never rewritten"


def test_replanning_clears_the_request_so_the_cycle_can_continue():
    merged = make_replanner(lambda **kw: _steps("b2"))(_state_with_a_failure())
    assert merged["replan_requested"] is False


def test_replanning_appends_a_revision_saying_why():
    merged = make_replanner(lambda **kw: _steps("b2"))(_state_with_a_failure())
    revision = merged["revisions"][0]
    assert revision["after_step"] == "a"
    assert "b2" in revision["change"]
    assert "line was dead" in revision["reason"]


def test_the_replanner_is_given_the_lessons():
    """Review of gap 3: a Lesson nothing reads is dead weight."""
    seen = {}

    def propose(*, goal, remaining, done, lessons):
        seen.update(lessons=lessons, remaining=[s["id"] for s in remaining],
                    done=[r["step_id"] for r in done], goal=goal)
        return _steps("b2")

    make_replanner(propose)(_state_with_a_failure())
    assert seen["goal"] == "g"
    assert seen["remaining"] == ["b", "c"], "only unfinished steps are up for revision"
    assert seen["done"] == ["a"]
    assert seen["lessons"][0]["failure"] == "line was dead"


def test_a_replan_that_proposes_nothing_leaves_no_work_and_ends_the_task():
    merged = make_replanner(lambda **kw: [])(_state_with_a_failure())
    assert merged["plan"] == []


def test_a_replan_may_legitimately_shorten_the_plan_below_what_is_done():
    """The case Review Focus 1 guards. Two steps are finished, the replanner
    returns one step, and next_step must still answer sensibly."""
    from lab.planning_agent.state import next_step

    state = _state_with_a_failure()
    state["done"] = [_result("a"), _result("b")]
    merged = make_replanner(lambda **kw: _steps("c"))(state)
    state["plan"] = merged["plan"]
    assert next_step(state)["id"] == "c"


def test_every_replan_counts_towards_the_cap():
    """Review Focus 7. The count is what a renamed step cannot evade."""
    merged = make_replanner(lambda **kw: _steps("b2"))(_state_with_a_failure())
    assert merged["replans"] == 1


def test_a_replan_that_raises_still_counts():
    def boom(**kw):
        raise RuntimeError("no")

    assert make_replanner(boom)(_state_with_a_failure())["replans"] == 1


def test_a_failing_proposer_leaves_the_plan_alone_and_records_the_failure():
    def boom(**kw):
        raise RuntimeError("the planner model refused")

    merged = make_replanner(boom)(_state_with_a_failure())
    assert "plan" not in merged, "a broken replanner must not wipe the plan"
    assert merged["replan_requested"] is False, "and must not loop forever"
    assert "refused" in merged["last_error"]
