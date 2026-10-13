r"""The topology: plan, then a bounded execute cycle, then one answer.

    START -> plan -> after_plan -> execute_step -> roll_up -> after_step -> execute_step
                          |                                        |  \-> replan -> execute_step
                          \-> synthesize -> END                    \-> synthesize -> END

Control flow is code. The nodes write state and these two router functions read
it, which is the rule `waku/graph/` already follows: models produce state,
functions pick the edge. Nothing here asks a model which way to go.

The node functions are injected so tests script them and the real client is
bound at the edge, the same shape as waku/graph/workflows/triage.py.

    pip install langgraph langgraph-checkpoint-sqlite
    python -m pytest lab/planning-agent/tests -q

SDK: langgraph 1.2.15. Verified 2026-10-12.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from lab.planning_agent.state import (
    MAX_REPLANS, PlanState, next_step, rolled_up_ids,
)

NodeFn = Callable[[PlanState], dict]


def _out_of_budget(state: PlanState) -> bool:
    return state["budget_spent"] >= state["budget_limit"]


def after_plan(state: PlanState) -> str:
    """Where to go once a plan exists.

    Both checks happen before any step runs. A limit of zero must not buy one
    step, and an empty plan must answer rather than spin the cycle.
    """
    if state["budget_limit"] <= 0 or _out_of_budget(state):
        return "synthesize"
    if next_step(state) is None:
        return "synthesize"
    return "step"


def after_step(state: PlanState) -> str:
    """Where to go once a step has finished.

    A requested replan comes first: it is set by a step that failed or that
    discovered work, and in both cases the remaining plan is not trustworthy.
    It does not come before the replan cap, though. A step that fails before
    its first model call costs nothing, so the budget cannot stop that cycle
    and MAX_REPLANS is what does.
    """
    if state["replan_requested"] and state["replans"] < MAX_REPLANS:
        return "replan"
    if _out_of_budget(state):
        return "synthesize"
    if next_step(state) is None:
        return "synthesize"
    return "step"


def build_graph(*, plan_fn: NodeFn, execute_fn: NodeFn, replan_fn: NodeFn,
                synthesize_fn: NodeFn, checkpointer=None):
    graph = StateGraph(PlanState)
    graph.add_node("plan", plan_fn)
    graph.add_node("execute_step", execute_fn)
    graph.add_node("replan", replan_fn)
    graph.add_node("synthesize", synthesize_fn)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", after_plan, {"step": "execute_step", "synthesize": "synthesize"})
    graph.add_node("roll_up", maybe_roll_up)
    graph.add_edge("execute_step", "roll_up")
    graph.add_conditional_edges(
        "roll_up", after_step,
        {"step": "execute_step", "replan": "replan", "synthesize": "synthesize"})
    # Conditional, not a plain edge. after_step answers "replan" before it
    # checks the budget, so a plain replan -> execute_step edge buys one step
    # past the limit. after_plan already checks the budget, an empty plan and
    # nothing-left, which is exactly the question being asked here.
    graph.add_conditional_edges(
        "replan", after_plan, {"step": "execute_step", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer)


ROLL_UP_EVERY = 8


def _results_not_yet_rolled_up(state: PlanState) -> list:
    covered = rolled_up_ids(state)
    return [r for r in state["done"] if r["step_id"] not in covered]


def maybe_roll_up(state: PlanState, every: int = ROLL_UP_EVERY) -> dict:
    """Collapse a finished phase once enough steps have piled up.

    A one-line digest per step still grows linearly, so past a couple of dozen
    steps the digest list alone crowds the prompt. The outputs stay on disk, so
    this shortens the prompt and loses nothing.
    """
    from lab.planning_agent.digest import make_rollup

    pending = _results_not_yet_rolled_up(state)
    if len(pending) < every:
        return {}
    return {"rollups": [make_rollup(pending[:every])]}
