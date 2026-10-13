import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.digest import ROLLUP_CHARS
from lab.planning_agent.graph import build_graph
from lab.planning_agent.idempotency import DedupTable
from lab.planning_agent.session import open_dedup, open_saver, thread_config
from lab.planning_agent.state import new_state


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def _app(saver, log, fail_on=None, world=None, table=None):
    """A graph whose step optionally commits a side effect and then crashes."""
    from lab.planning_agent.state import next_step

    def plan_fn(state):
        return {"plan": _steps("a", "b", "c")}

    def execute_fn(state):
        step = next_step(state)

        def act():
            if world is not None:
                world.append(step["id"])
            return step["id"]

        if table is None:
            act()
        else:
            table.guard("t1", step["id"], "create_event", {"n": step["id"]}, act)
        log.append(step["id"])
        if fail_on and step["id"] == fail_on[0] and fail_on[1]["armed"]:
            raise RuntimeError(f"crash in {step['id']}")
        return {"done": [{"step_id": step["id"], "digest": "", "artifacts": {},
                          "output_path": ""}],
                "budget_spent": state["budget_spent"] + 0.01}

    return build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                       replan_fn=lambda s: {}, synthesize_fn=lambda s: {},
                       checkpointer=saver)


def test_a_finished_task_needs_no_resume(tmp_path):
    saver, conn = open_saver(tmp_path / "ckpt.sqlite")
    log: list[str] = []
    out = _app(saver, log).invoke(new_state("g", 10.0), thread_config("t1"))
    assert log == ["a", "b", "c"]
    assert [r["step_id"] for r in out["done"]] == ["a", "b", "c"]
    conn.close()


def test_a_crashed_task_resumes_in_a_fresh_process_without_redoing_work(tmp_path):
    """The claim the whole topic rests on. The second half uses a brand-new
    connection, which is what a restarted process has."""
    db = tmp_path / "ckpt.sqlite"
    armed = {"armed": True}

    saver1, conn1 = open_saver(db)
    log1: list[str] = []
    with pytest.raises(RuntimeError):
        _app(saver1, log1, fail_on=("b", armed)).invoke(
            new_state("g", 10.0), thread_config("t1"))
    assert log1 == ["a", "b"]
    conn1.close()

    armed["armed"] = False               # the transient cause is gone
    saver2, conn2 = open_saver(db)       # a fresh process
    log2: list[str] = []
    app = _app(saver2, log2, fail_on=("b", armed))
    assert [r["step_id"] for r in app.get_state(thread_config("t1")).values["done"]] == ["a"]
    out = app.invoke(None, thread_config("t1"))
    assert log2 == ["b", "c"], "'a' must not run again"
    assert [r["step_id"] for r in out["done"]] == ["a", "b", "c"]
    conn2.close()


def test_resume_repeats_a_side_effect_unless_the_dedup_table_guards_it(tmp_path):
    """Review of the whole design in one test. The checkpoint stays consistent
    either way; only the dedup table keeps the world consistent."""
    for use_table, expected in ((False, ["a", "b", "b", "c"]), (True, ["a", "b", "c"])):
        db = tmp_path / f"ckpt-{use_table}.sqlite"
        armed = {"armed": True}
        world: list[str] = []
        table = None
        if use_table:
            table, dedup_conn = open_dedup(tmp_path / f"state-{use_table}.db")

        saver1, conn1 = open_saver(db)
        with pytest.raises(RuntimeError):
            _app(saver1, [], fail_on=("b", armed), world=world, table=table).invoke(
                new_state("g", 10.0), thread_config("t1"))
        conn1.close()

        armed["armed"] = False
        saver2, conn2 = open_saver(db)
        _app(saver2, [], fail_on=("b", armed), world=world, table=table).invoke(
            None, thread_config("t1"))
        conn2.close()
        if use_table:
            dedup_conn.close()

        assert world == expected, f"use_table={use_table}"


def test_the_checkpoint_does_not_grow_with_the_number_of_finished_steps(tmp_path):
    """Why full outputs go to disk. A 4KB output per step carried in state
    would be re-serialised on every later step."""
    db = tmp_path / "ckpt.sqlite"
    saver, conn = open_saver(db)
    from lab.planning_agent.state import next_step

    def plan_fn(state):
        return {"plan": _steps(*[f"s{i}" for i in range(12)])}

    def execute_fn(state):
        step = next_step(state)
        return {"done": [{"step_id": step["id"], "digest": "d" * 200,
                          "artifacts": {}, "output_path": "/tmp/x"}],
                "budget_spent": state["budget_spent"] + 0.01}

    app = build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                      replan_fn=lambda s: {}, synthesize_fn=lambda s: {},
                      checkpointer=saver)
    app.invoke(new_state("g", 10.0), thread_config("t1"))
    sizes = [len(str(snap.values)) for snap in app.get_state_history(thread_config("t1"))]
    conn.close()
    growth = max(sizes) - min(sizes)
    # What state is allowed to gain over the run: one 200-char digest per step
    # with room for the record around it, the 12-step plan, and the one rollup
    # the roll_up node collapses at eight steps. A 4KB step output riding in
    # state would be 48,000, so this still catches the thing it guards.
    allowance = 12 * 400 + ROLLUP_CHARS + 200
    assert growth < allowance, (
        "state grew faster than one 200-char digest per step; something large "
        f"is riding in state (growth={growth}, allowance={allowance})")


def test_waku_create_event_reports_already_exists_on_a_replay(tmp_path):
    """Why the planner keys its own calls rather than trusting the tool.

    calendar.py guards by natural key, so a replay is told the event exists.
    That is correct for the local table, and it is also why a replay cannot
    tell 'nothing to do' from 'the downstream syncs never ran'.
    """
    from waku import db
    from waku.tools import calendar

    conn = db.connect(tmp_path)
    tool = calendar.make_tool(conn, tmp_path)
    first = tool.fn(title="lunch", start="2026-10-12T12:00")
    second = tool.fn(title="lunch", start="2026-10-12T12:00")
    conn.close()

    assert "already exists" not in first
    assert "already exists" in second
    assert "not duplicated" in second


WORKER = '''
import os, sqlite3, sys
sys.path.insert(0, {root!r})
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_pa", os.path.join({root!r}, "lab", "planning-agent", "_path.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

from lab.planning_agent.digest import ROLLUP_CHARS
from lab.planning_agent.graph import build_graph
from lab.planning_agent.idempotency import DedupTable
from lab.planning_agent.session import open_saver, thread_config
from lab.planning_agent.state import new_state, next_step

ckpt, state_db, world, die = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "die"
saver, conn = open_saver(ckpt)
dedup_conn = sqlite3.connect(state_db)
table = DedupTable(dedup_conn); table.setup()

def plan_fn(state):
    return {{"plan": [{{"id": i, "description": i, "status": "pending"}}
                     for i in ("a", "b", "c")]}}

def execute_fn(state):
    step = next_step(state)
    def act():
        with open(world, "a", encoding="utf-8") as fh:
            fh.write(step["id"] + "\\n")
        return step["id"]
    table.guard("t1", step["id"], "create_event", {{"n": step["id"]}}, act)
    if die and step["id"] == "b":
        # No exception, no unwinding, no checkpoint: the process is simply gone,
        # which is what a kill -9 or a closed laptop lid does.
        os._exit(1)
    return {{"done": [{{"step_id": step["id"], "digest": "", "artifacts": {{}},
                      "output_path": ""}}],
            "budget_spent": state["budget_spent"] + 0.01}}

app = build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                  replan_fn=lambda s: {{}}, synthesize_fn=lambda s: {{}},
                  checkpointer=saver)
app.invoke(new_state("g", 10.0) if sys.argv[5] == "start" else None,
           thread_config("t1"))
conn.close(); dedup_conn.close()
'''


def test_a_killed_process_resumes_without_repeating_the_side_effect(tmp_path):
    """The claim the README makes, tested the way the README words it. The
    first process does not raise: it calls os._exit(1) after the side effect has
    committed and before the checkpoint does, so nothing unwinds."""
    root = str(Path(__file__).resolve().parents[3])
    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent(WORKER).format(root=root), encoding="utf-8")
    ckpt, state_db, world = (str(tmp_path / "ckpt.sqlite"),
                             str(tmp_path / "state.db"), str(tmp_path / "world.txt"))

    first = subprocess.run([sys.executable, str(worker), ckpt, state_db, world,
                            "die", "start"], capture_output=True, text=True)
    assert first.returncode == 1, f"the worker should have died: {first.stderr}"
    assert Path(world).read_text().split() == ["a", "b"]

    second = subprocess.run([sys.executable, str(worker), ckpt, state_db, world,
                             "live", "resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert Path(world).read_text().split() == ["a", "b", "c"], (
        "'b' committed before the process died and must not commit again")
