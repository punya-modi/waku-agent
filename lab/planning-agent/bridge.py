"""One planned step is one Waku loop turn.

`run_loop` is not modified and not wrapped in anything clever: the step builds
a prompt, calls it with a scoped registry, and keeps what came back. This is
the same move waku/graph/nodes.agent_node makes, and it is what makes this a
Waku experiment rather than a LangGraph tutorial.

What the step reads is the point. It gets the goal, the plan, the rollups of
finished phases, the digests and artifacts of recent steps, and any lessons.
It never gets an earlier step's transcript: that is on disk, and the step can
ask for it.
"""

from __future__ import annotations

from collections.abc import Callable

from waku.loop.trim import shrink_seen

from lab.planning_agent.digest import step_digest
from lab.planning_agent.offload import Scratch
from lab.planning_agent.state import Lesson, PlanState, Step, StepResult, next_step

ARTIFACTS_MARKER = "ARTIFACTS:"

PROMPT = """\
You are working through one step of a larger task.

The overall goal, which never changes:
{goal}

The plan as it stands:
{plan}
{history}
Your step now: {step}

When your step produces identifiers that a later step will need exactly - an
event id, a file path, a URL - end your reply with one line:
{marker} key=value; key=value

Do not restate the plan. Report what you did and what you found."""


class UsageMeter:
    """Turns the loop's llm events into dollars.

    run_loop reports token counts per call through its observer; a receipt is
    assembled higher up, in Waku.respond, which a lab run does not go through.
    Prices are USD per million tokens, the unit providers quote.
    """

    def __init__(self, price_in: float, price_out: float) -> None:
        self.price_in = price_in
        self.price_out = price_out
        self.cost = 0.0

    def __call__(self, kind: str, event: dict) -> None:
        if kind != "llm":
            return
        usage = event.get("usage") or {}
        self.cost += (usage.get("in", 0) / 1_000_000) * self.price_in
        self.cost += (usage.get("out", 0) / 1_000_000) * self.price_out


def parse_artifacts(reply: str) -> dict[str, str]:
    """Identifiers from the reply's marker line, kept exactly as written.

    A pair without an '=' is skipped rather than fatal: the line is written by
    a model, so it is input, and a malformed half must not lose the good half.
    """
    for line in reversed(reply.splitlines()):
        if not line.strip().startswith(ARTIFACTS_MARKER):
            continue
        body = line.strip()[len(ARTIFACTS_MARKER):]
        found: dict[str, str] = {}
        for pair in body.split(";"):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            if key.strip():
                found[key.strip()] = value.strip()
        return found
    return {}


def _history(state: PlanState) -> str:
    """What has happened, with anything already summarised left out.

    A rolled-up step's digest must not also appear on its own, or the rollup
    adds to the prompt instead of shortening it. `done` is append-only, so a
    rollup cannot remove anything from it; filtering here is how the collapse
    actually reaches the prompt.
    """
    rolled_up = {sid for rollup in state["rollups"] for sid in rollup["step_ids"]}
    lines: list[str] = []
    for rollup in state["rollups"]:
        lines.append(f"- earlier phase ({', '.join(rollup['step_ids'])}): {rollup['summary']}")
    for result in state["done"]:
        if result["step_id"] in rolled_up:
            continue
        artifacts = "; ".join(f"{k}={v}" for k, v in result["artifacts"].items())
        lines.append(f"- {result['step_id']}: {result['digest']}"
                     + (f" [{artifacts}]" if artifacts else ""))
    for lesson in state["lessons"]:
        lines.append(f"- lesson from {lesson['step_id']}: tried {lesson['attempt']}, "
                     f"{lesson['failure']}. {lesson['diagnosis']}")
    if not lines:
        return ""
    return "\nWhat has happened so far:\n" + "\n".join(lines) + "\n"


def step_prompt(state: PlanState, step: Step) -> str:
    plan = "\n".join(
        f"{i}. [{s['status']}] {s['description']}" for i, s in enumerate(state["plan"], 1))
    return PROMPT.format(goal=state["goal"], plan=plan, history=_history(state),
                         step=step["description"], marker=ARTIFACTS_MARKER)


# `trim=shrink_seen` is passed to run_loop below. It is the first of the four
# memory mechanisms and the only one that already exists in waku: inside a
# single step, a tool result the model has read is cut before the next call.
# The other three mechanisms cover the span between steps, which this one
# cannot see.


def make_execute_step(*, run_loop: Callable, client, model: str, system: str,
                      tools, scratch: Scratch, thread_id: str,
                      price_in: float, price_out: float, dedup=None,
                      max_iterations: int = 10, max_tokens: int = 2048) -> Callable:
    """The execute_step node, bound to one client, registry and scratch."""

    def execute(state: PlanState) -> dict:
        step = next_step(state)
        if step is None:
            return {}

        scoped = tools
        if dedup is not None:
            from lab.planning_agent.idempotency import guarded_registry
            scoped = guarded_registry(tools, dedup, thread_id, step["id"])

        meter = UsageMeter(price_in=price_in, price_out=price_out)
        messages = [{"role": "user", "content": step_prompt(state, step)}]
        try:
            result = run_loop(client=client, model=model, system=system,
                              messages=messages, tools=scoped,
                              max_iterations=max_iterations, max_tokens=max_tokens,
                              observer=meter, trim=shrink_seen)
        except Exception as exc:
            # Surface, do not crash: the graph decides what happens next, and a
            # lesson is what stops the replanner proposing this again. `done`
            # is appended to, so the empty list says "this step produced
            # nothing" without touching the history.
            return {"replan_requested": True,
                    "done": [],
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "budget_spent": state["budget_spent"] + meter.cost,
                    "lessons": [Lesson(step_id=step["id"],
                                       attempt=step["description"],
                                       failure=f"{type(exc).__name__}: {exc}",
                                       diagnosis="the step did not complete; "
                                                 "try a different approach")]}

        output_path = scratch.write(step["id"], result.reply)
        return {"done": [StepResult(step_id=step["id"],
                                    digest=step_digest(result.reply),
                                    artifacts=parse_artifacts(result.reply),
                                    output_path=output_path)],
                "budget_spent": state["budget_spent"] + meter.cost,
                "replan_requested": False}

    return execute
