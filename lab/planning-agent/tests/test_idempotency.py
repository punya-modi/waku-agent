import operator
import sqlite3
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from lab.planning_agent.idempotency import (
    DedupTable, call_key, is_side_effecting,
)


def test_a_tool_that_is_not_named_read_only_is_treated_as_side_effecting():
    """The guard is a closed set: read-only tools are named, and anything
    unknown is assumed to change the world. Guessing the other way round means
    a new tool silently skips the dedup table."""
    assert is_side_effecting("create_event") is True
    assert is_side_effecting("send_message") is True
    assert is_side_effecting("a_tool_nobody_has_written_yet") is True
    assert is_side_effecting("list_events") is False


def test_the_key_changes_with_every_part_of_the_call():
    base = ("t1", "s1", "create_event", {"title": "lunch"})
    assert call_key(*base) == call_key("t1", "s1", "create_event", {"title": "lunch"})
    assert call_key(*base) != call_key("t2", "s1", "create_event", {"title": "lunch"})
    assert call_key(*base) != call_key("t1", "s2", "create_event", {"title": "lunch"})
    assert call_key(*base) != call_key("t1", "s1", "send_message", {"title": "lunch"})
    assert call_key(*base) != call_key("t1", "s1", "create_event", {"title": "dinner"})


def test_the_key_ignores_argument_order():
    assert call_key("t", "s", "x", {"a": 1, "b": 2}) == call_key("t", "s", "x", {"b": 2, "a": 1})


def test_two_tasks_doing_the_same_thing_do_not_dedupe_against_each_other():
    """Review Focus 3. Without thread_id in the key, the second task's real
    booking is silently skipped."""
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    calls = []

    def book():
        calls.append("booked")
        return "booked"

    table.guard("task-1", "s1", "create_event", {"title": "lunch"}, book)
    table.guard("task-2", "s1", "create_event", {"title": "lunch"}, book)
    assert calls == ["booked", "booked"], "two separate tasks must both act"


def test_a_failed_call_is_recorded_but_not_replayed():
    """Review Focus 8. A timeout is transient. Recording it keeps the crash
    window closed; replaying it would freeze one bad minute into the task."""
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    results = iter(["Error running create_event: timed out", "event 7 created"])

    def book():
        return next(results)

    first = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    second = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    assert first.startswith("Error running")
    assert second == "event 7 created", "the retry must reach the tool"
    rows = conn.execute("SELECT status FROM planning_tool_calls").fetchall()
    assert [r[0] for r in rows] == ["ok"], "the error row was replaced by the success"


def test_a_raised_call_is_recorded_as_an_error_and_re_raised():
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()

    def boom():
        raise RuntimeError("the API refused")

    with pytest.raises(RuntimeError, match="refused"):
        table.guard("t", "s1", "create_event", {"title": "lunch"}, boom)
    row = conn.execute("SELECT status, result FROM planning_tool_calls").fetchone()
    assert row[0] == "error"
    assert "refused" in row[1]


def test_setup_refuses_a_table_that_predates_the_status_column():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE planning_tool_calls (key TEXT PRIMARY KEY, result TEXT)")
    conn.commit()
    with pytest.raises(RuntimeError, match="status column"):
        DedupTable(conn).setup()


def test_a_replayed_call_returns_the_recorded_result_without_running_again():
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    calls = []

    def book():
        calls.append("ran")
        return "event 7 created"

    first = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    second = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    assert first == second == "event 7 created"
    assert calls == ["ran"], "the second call must not reach the tool"


class _S(TypedDict):
    plan: list[str]
    done: Annotated[list[str], operator.add]
    fail_once: bool


def _build(saver, world, table):
    """A graph whose step commits a side effect and may then crash."""

    def plan(state):
        return {"plan": ["a", "b", "c"]}

    def step(state):
        finished = set(state["done"])
        nxt = next(s for s in state["plan"] if s not in finished)

        def act():
            world.append(nxt)
            return nxt

        if table is None:
            act()
        else:
            table.guard("t1", nxt, "create_event", {"name": nxt}, act)
        if state.get("fail_once") and nxt == "b":
            raise RuntimeError("crash after the side effect committed")
        return {"done": [nxt]}

    def more(state):
        return "step" if len(state["done"]) < len(state["plan"]) else "done"

    graph = StateGraph(_S)
    graph.add_node("plan", plan)
    graph.add_node("step", step)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "step")
    graph.add_conditional_edges("step", more, {"step": "step", "done": END})
    return graph.compile(checkpointer=saver)


def _run_to_completion(tmp_path, table):
    """Crash mid-step, then resume, and report what the world saw."""
    conn = sqlite3.connect(tmp_path / "ckpt.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    world: list[str] = []
    config = {"configurable": {"thread_id": "t1"}}
    app = _build(saver, world, table)
    with pytest.raises(RuntimeError):
        app.invoke({"plan": [], "done": [], "fail_once": True}, config)
    app.update_state(config, {"fail_once": False})  # the transient cause is gone
    out = app.invoke(None, config)
    conn.close()
    return world, out["done"]


def test_without_the_dedup_table_resume_repeats_the_side_effect(tmp_path):
    """The failure this module exists to prevent, demonstrated. The
    checkpointed state is clean; the outside world is not."""
    world, done = _run_to_completion(tmp_path, table=None)
    assert done == ["a", "b", "c"], "state stays consistent"
    assert world == ["a", "b", "b", "c"], "but 'b' was committed twice"


def test_with_the_dedup_table_resume_does_not_repeat_the_side_effect(tmp_path):
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    world, done = _run_to_completion(tmp_path, table=table)
    assert done == ["a", "b", "c"]
    assert world == ["a", "b", "c"], "'b' must be committed exactly once"
