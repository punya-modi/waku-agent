"""DETERMINISTIC EVAL — the plan graph: two approval gates, never one per Step.

The planning agent pauses for the user at exactly two points: after the first
Plan, and after a Revision. These checks drive the bare topology with harmless
stub callables and LangGraph's own `.invoke()` / `Command(resume=...)`, so they
prove the routing and the pauses with no model, no network and no tools.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys

import pytest

pytest.importorskip("langgraph", reason="the planning agent is the [planning] extra")

from langgraph.types import Command  # noqa: E402

from waku.graph.workflows import plan as plan_module  # noqa: E402
from waku.graph.workflows.plan import build_plan_graph  # noqa: E402

THREAD = {"configurable": {"thread_id": "t1"}}


def _steps(*ids):
    return [{"id": i, "description": f"do {i}"} for i in ids]


def _graph(*, steps=("a", "b"), replan_on=(), revised=("x",)):
    """A plan graph over stubs. `calls` records which node ran, in order."""
    calls: list[str] = []

    def plan_fn(state):
        calls.append("plan")
        return {"plan": _steps(*steps)}

    def execute_fn(state):
        step = plan_module.next_step(state)
        calls.append(f"step:{step['id']}")
        return {
            "done": [{"step_id": step["id"], "digest": "ok"}],
            "replan_requested": step["id"] in replan_on,
        }

    def replan_fn(state):
        calls.append("replan")
        finished = [s for s in state["plan"] if s["id"] in {d["step_id"] for d in state["done"]}]
        return {
            "plan": finished + _steps(*revised),
            "revisions": [{"after_step": state["done"][-1]["step_id"], "change": "new tail"}],
        }

    def synthesize_fn(state):
        calls.append("synthesize")
        return {"digest": "DONE"}

    graph = build_plan_graph(
        plan_fn=plan_fn, execute_fn=execute_fn,
        replan_fn=replan_fn, synthesize_fn=synthesize_fn,
    )
    return graph, calls


def _start(graph):
    return graph.invoke({"goal": "g", "done": [], "lessons": [], "revisions": [],
                         "replan_requested": False}, THREAD)


def _approve(graph):
    return graph.invoke(Command(resume="approve"), THREAD)


def test_pauses_after_the_plan_before_any_step_runs():
    graph, calls = _graph()
    result = _start(graph)
    assert calls == ["plan"]
    assert result["__interrupt__"]
    assert [s["id"] for s in graph.get_state(THREAD).values["plan"]] == ["a", "b"]


def test_approving_runs_every_step_without_pausing_between_them():
    graph, calls = _graph()
    _start(graph)
    result = _approve(graph)
    assert calls == ["plan", "step:a", "step:b", "synthesize"]
    assert "__interrupt__" not in result
    assert result["digest"] == "DONE"


def test_approving_resumes_rather_than_restarting():
    """The plan callable ran once: resume continues from the pause."""
    graph, calls = _graph()
    _start(graph)
    _approve(graph)
    assert calls.count("plan") == 1


def test_a_step_requesting_a_revision_routes_to_replan_and_pauses_again():
    graph, calls = _graph(replan_on=("a",))
    _start(graph)
    paused = _approve(graph)
    assert calls == ["plan", "step:a", "replan"]
    assert paused["__interrupt__"]


def test_after_the_revision_is_approved_the_new_tail_runs():
    graph, calls = _graph(replan_on=("a",), revised=("x", "y"))
    _start(graph)
    _approve(graph)
    result = _approve(graph)
    assert calls == ["plan", "step:a", "replan", "step:x", "step:y", "synthesize"]
    assert "__interrupt__" not in result
    assert [r["after_step"] for r in result["revisions"]] == ["a"]


def test_replan_does_not_fire_when_no_step_asks_for_it():
    graph, calls = _graph()
    _start(graph)
    _approve(graph)
    assert "replan" not in calls


def test_the_gate_fires_twice_at_most_not_once_per_step():
    graph, _ = _graph(steps=("a", "b", "c", "d"), replan_on=("a",), revised=("x", "y", "z"))
    pauses = 0
    result = _start(graph)
    while "__interrupt__" in result:
        pauses += 1
        result = _approve(graph)
    assert pauses == 2


def test_an_empty_plan_goes_straight_to_synthesize_after_approval():
    graph, calls = _graph(steps=())
    _start(graph)
    _approve(graph)
    assert calls == ["plan", "synthesize"]


def test_the_module_imports_only_what_a_pure_topology_needs():
    """Pure topology: everything real is injected by the binder. A closed set,
    so a new import (a model client, pathlib, os) fails until it is allowed
    here on purpose."""
    allowed = {"__future__", "operator", "collections", "typing", "langgraph"}
    tree = ast.parse(inspect.getsource(plan_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= allowed, imported - allowed


def test_a_missing_extra_fails_with_an_install_hint(monkeypatch):
    """Without langgraph the import says what to install, not a bare crash."""
    for name in [m for m in sys.modules if m == "langgraph" or m.startswith("langgraph.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "langgraph", None)  # makes `import langgraph` fail
    monkeypatch.delitem(sys.modules, "waku.graph.workflows.plan")
    with pytest.raises(ImportError, match=r"waku-agent\[planning\]"):
        importlib.import_module("waku.graph.workflows.plan")
