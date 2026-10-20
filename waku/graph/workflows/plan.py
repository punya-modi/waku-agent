"""The plan graph workflow — a goal, a Plan the user approves, then the Steps.

    START -> plan -> approve_plan -> execute_step -+-> execute_step   (unfinished Steps)
                                                   +-> replan -> approve_revision -> execute_step
                                                   +-> synthesize -> END

Control flow is code: nodes write state, the routers below read it, and nothing
asks a model which way to go. The four node functions are injected, so this
module holds no client and no filesystem reference and a test scripts it for
free. That injection pattern is triage.py's; the engine underneath is not:
triage.py runs on waku.graph.engine, this runs on LangGraph (ADR-0001).

The Approval gate (see CONTEXT.md) is LangGraph's interrupt(): the run stops
and waits for Command(resume=...). It fires after `plan` and after `replan`,
never after a single Step. Each gate is its own node because LangGraph re-runs
a node from the top on resume: an interrupt() inside `plan` would call the
model again and could execute a different Plan from the one the user approved.

Needs the [planning] extra (langgraph). interrupt() only works with a
checkpointer, so the default is the in-memory one: a paused run lives as long
as the process does. A binder may pass its own `checkpointer`.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, TypedDict

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt
except ImportError as exc:
    raise ImportError(
        "The planning agent needs the 'planning' extra — "
        "pip install 'waku-agent[planning]' (in a checkout: pip install -e '.[planning]')"
    ) from exc

NodeFn = Callable[[dict], dict]


class PlanState(TypedDict, total=False):
    goal: str  # never changes once planning starts
    plan: list[dict]  # the current truth: replaced outright, never patched
    done: Annotated[list[dict], operator.add]  # one {step_id, digest} per Step run
    lessons: Annotated[list[dict], operator.add]
    revisions: Annotated[list[dict], operator.add]
    replan_requested: bool
    digest: str


def next_step(state: dict) -> dict | None:
    """The first Step in the Plan with no entry in `done`, or None.

    A Step that failed has an entry too: it is finished, and the Revision
    decides what replaces the rest.
    """
    finished = {d["step_id"] for d in state.get("done", [])}
    return next((s for s in state.get("plan", []) if s["id"] not in finished), None)


def after_approval(state: dict) -> str:
    return "step" if next_step(state) else "synthesize"


def after_step(state: dict) -> str:
    """A requested Revision comes first: the remaining Plan is not trustworthy."""
    if state.get("replan_requested"):
        return "replan"
    return "step" if next_step(state) else "synthesize"


def _gate(state: dict) -> dict:
    """Stop and wait for the user. Whatever they resume with means go."""
    interrupt({"plan": state.get("plan", [])})
    return {}


def build_plan_graph(*, plan_fn: NodeFn, execute_fn: NodeFn, replan_fn: NodeFn,
                     synthesize_fn: NodeFn, checkpointer=None):
    def replan(state: dict) -> dict:
        return {**replan_fn(state), "replan_requested": False}

    graph = StateGraph(PlanState)
    graph.add_node("plan", plan_fn)
    graph.add_node("approve_plan", _gate)
    graph.add_node("execute_step", execute_fn)
    graph.add_node("replan", replan)
    graph.add_node("approve_revision", _gate)
    graph.add_node("synthesize", synthesize_fn)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "approve_plan")
    graph.add_conditional_edges(
        "approve_plan", after_approval, {"step": "execute_step", "synthesize": "synthesize"})
    graph.add_conditional_edges(
        "execute_step", after_step,
        {"step": "execute_step", "replan": "replan", "synthesize": "synthesize"})
    graph.add_edge("replan", "approve_revision")
    graph.add_conditional_edges(
        "approve_revision", after_approval, {"step": "execute_step", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
