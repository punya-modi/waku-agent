"""The task's state: one goal, a plan that is replaced, and histories that grow.

The reducers are the design. `plan` carries no reducer, so each write replaces
it outright: it is always the current truth, short enough to recite into every
step's prompt. `done`, `revisions`, `lessons` and `rollups` use operator.add,
so they accumulate: they are the task's event history, and nothing in them is
ever rewritten.

`goal` never changes. It is re-injected on every step, which is what stops a
long task drifting off the original ask.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Step(TypedDict):
    id: str
    description: str
    status: str  # pending | in_progress | done | failed


class StepResult(TypedDict):
    step_id: str
    digest: str                # the short form the next step reads
    artifacts: dict[str, str]  # identifiers, verbatim
    output_path: str           # the full output, on disk


class Revision(TypedDict):
    after_step: str
    change: str
    reason: str


class Lesson(TypedDict):
    step_id: str
    attempt: str
    failure: str
    diagnosis: str


class Rollup(TypedDict):
    step_ids: list[str]
    summary: str


class PlanState(TypedDict):
    goal: str
    plan: list[Step]
    done: Annotated[list[StepResult], operator.add]
    revisions: Annotated[list[Revision], operator.add]
    lessons: Annotated[list[Lesson], operator.add]
    rollups: Annotated[list[Rollup], operator.add]
    budget_spent: float  # USD, the same unit as the usage.jsonl ledger
    budget_limit: float
    replan_requested: bool
    last_error: str
    # Replans so far, across the whole task. A bare annotation, so each write
    # replaces it. This is the termination bound that cannot be evaded: the
    # per-step one below counts lessons by step id, and a replanner that
    # renames a failed step resets that count to zero.
    replans: int


# Two attempts at a step, then it is skipped rather than retried for ever.
MAX_STEP_ATTEMPTS = 2
# And five replans for the whole task, whatever the step ids look like.
MAX_REPLANS = 5


def new_state(goal: str, budget_limit: float) -> PlanState:
    return PlanState(goal=goal, plan=[], done=[], revisions=[], lessons=[],
                     rollups=[], budget_spent=0.0, budget_limit=budget_limit,
                     replan_requested=False, last_error="", replans=0)


def attempts(state: PlanState, step_id: str) -> int:
    """How many times a step has been tried and left a lesson behind."""
    return sum(1 for lesson in state["lessons"] if lesson["step_id"] == step_id)


def next_step(state: PlanState) -> Step | None:
    """The first step in `plan` that is neither finished nor exhausted, or None.

    Selected by id, never by position. A replanner may reorder the plan, drop
    steps from it or shorten it, and `len(done)` can legitimately exceed
    `len(plan)` afterwards; an index would then point past the end or at a step
    that already ran.

    A step with MAX_STEP_ATTEMPTS lessons against it is skipped. Without that,
    a step that fails before its first model call costs nothing, leaves
    `budget_spent` untouched, and is selected again for ever: the budget is not
    a termination bound when failing is free.
    """
    finished = {result["step_id"] for result in state["done"]}
    for step in state["plan"]:
        if step["id"] in finished:
            continue
        if attempts(state, step["id"]) >= MAX_STEP_ATTEMPTS:
            continue
        return step
    return None
