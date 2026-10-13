"""Revising the plan without losing what already happened.

Roll back the work, keep the reflection. A failed step's output is discarded;
the lesson it produced is not, and the replanner is handed those lessons along
with the unfinished steps. Without that it is invoked at exactly the moment
nothing records what went wrong, and it can propose the failing approach again.

`done` is append-only, so this function cannot rewrite history even by mistake.
It returns a new `plan` and one `Revision`, and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable

from lab.planning_agent.state import PlanState, Revision, Step, StepResult

Propose = Callable[..., list[Step]]


def preserved(state: PlanState) -> list[StepResult]:
    """What the replanner may not touch."""
    return list(state["done"])


def _remaining(state: PlanState) -> list[Step]:
    finished = {result["step_id"] for result in state["done"]}
    return [step for step in state["plan"] if step["id"] not in finished]


def make_replanner(propose: Propose) -> Callable[[PlanState], dict]:
    """A replan node. `propose` returns the steps that should replace the
    unfinished ones; it never sees a chance to change the finished ones."""

    def replan(state: PlanState) -> dict:
        done = preserved(state)
        last = done[-1]["step_id"] if done else ""
        try:
            steps = propose(goal=state["goal"], remaining=_remaining(state),
                            done=done, lessons=list(state["lessons"]))
        except Exception as exc:
            # A broken replanner must not wipe the plan and must not leave the
            # request set, which would route straight back here for ever. The
            # attempt still counts: a replanner that raises every time is
            # exactly what MAX_REPLANS exists to stop.
            return {"replan_requested": False,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "replans": state["replans"] + 1}

        return {"plan": list(steps),
                "replan_requested": False,
                "last_error": "",
                # Counted here, not in the router: this is where a replan
                # actually happened. after_step reads it and stops at
                # MAX_REPLANS, which is the bound a replanner cannot evade by
                # renaming the step that keeps failing.
                "replans": state["replans"] + 1,
                "revisions": [Revision(
                    after_step=last,
                    change="remaining steps are now "
                           + (", ".join(s["id"] for s in steps) or "(none)"),
                    reason=state["last_error"] or "new work was discovered")]}

    return replan
