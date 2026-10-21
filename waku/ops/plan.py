"""`/plan <goal>` — break a goal into Steps, approve the Plan, then run them.

    /plan book a focus block every weekday morning   -> shows the Plan, pauses
    /plan approve                                    -> runs every Step for real
    /plan stop                                       -> drops the paused Plan

This is the BINDER for waku/graph/workflows/plan.py: the pure graph takes four
callables, and this is where they meet a real model and the real ToolRegistry.
Each Step is one run_loop turn, the same move waku/graph/nodes.py:agent_node
makes. The pause is the graph's own (LangGraph interrupt); this module only
remembers which run is paused.

One dashboard, one user, one run at a time: the paused run lives in the single
module-level `_paused` slot, and a process restart loses it (Phase 1, on purpose).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass

from langgraph.types import Command

from waku.graph.workflows.plan import build_plan_graph, next_step
from waku.loop.agent import error_text, run_loop
from waku.ops.browser_agent import agent_lock, get_agent

USAGE = ("`/plan` needs a goal.\n\n"
         "Try `/plan look at my calendar this week and book a 30-minute focus block "
         "each weekday morning`. I will show you the Plan and wait for `/plan approve`.")

PLAN_PROMPT = """\
Break this goal into a short, ordered list of Steps. Each Step is one unit of
work an assistant with tools (calendar, notes, memory, search) can do in a
single turn. Use as few Steps as the goal needs.

Goal: {goal}

Reply with JSON only: {{"steps": ["first step", "second step"]}}"""

STEP_PROMPT = """\
You are working through one step of a larger task.

The overall goal, which never changes:
{goal}

The plan as it stands:
{plan}
{history}
Your step now: {step}

Do only this step. Report what you did and what you found; do not restate the plan."""

SYNTH_PROMPT = """\
You ran a plan for the user. Write the final reply: what was done, what failed
(if anything), and anything they should check. Be brief and concrete.

Goal: {goal}

What happened:
{history}"""


# What a Step may do on its own. Everything else (send_message, delegate_task,
# create_skill, update_soul, manage_memory, any MCP or Apple tool) is an
# Escalation: refused inside the Step, queued, and run only if the user approves
# the next gate. A new tool is therefore refused until someone adds it here.
ALLOWED_TOOLS = frozenset({"list_events", "create_event", "save_note", "search_web",
                           "github_read"})


class _ScopedTools:
    """The real registry with a fixed allowlist in front of `execute`.

    The model still SEES every schema, so an out-of-list call can be attempted
    and caught here; hiding them would only make an injected call fail as an
    unknown tool and leave the user no way to approve it. `queued` collects the
    refused calls for the Step to hand to the graph.
    """

    def __init__(self, real, step_id: str, first_id: int) -> None:
        self._real, self._step_id, self._next = real, step_id, first_id
        self.queued: list[dict] = []

    def schemas(self) -> list:
        return self._real.schemas()

    def execute(self, name: str, args: dict, notify=None) -> str:
        if name in ALLOWED_TOOLS:
            return self._real.execute(name, args, notify=notify)
        self.queued.append({"id": f"e{self._next + len(self.queued)}", "step_id": self._step_id,
                            "tool": name, "args": args})
        return (f"Not run: `{name}` is outside what a plan may do on its own. It is queued "
                "for the user's approval. Carry on without it and say what you needed.")


@dataclass
class _Run:
    """One paused run: the compiled graph, its thread, and who is watching now."""
    app: object
    thread_id: str
    waku: object
    observer: object = None

    def notify(self, kind: str, ev: dict) -> None:
        """Trace the event and pass it to whoever is watching THIS call: a resume
        arrives on a new request with a new observer."""
        self.waku.tracer.event(kind, ev)
        if self.observer:
            self.observer(kind, ev)


_paused: _Run | None = None


def _steps_from(text: str) -> list[dict]:
    """Steps from the planner's reply. The reply is model output, so it is
    input: tolerate a code fence, reject anything that is not a non-empty
    list of strings."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        raw = json.loads(match.group(0))["steps"] if match else None
    except (ValueError, KeyError, TypeError):
        raw = None
    if not isinstance(raw, list) or not all(isinstance(s, str) and s.strip() for s in raw) or not raw:
        raise ValueError("the planner did not return a list of Steps")
    return [{"id": f"s{i}", "description": s.strip()} for i, s in enumerate(raw, 1)]


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _plan_lines(state: dict) -> str:
    finished = {d["step_id"] for d in state.get("done", [])}
    return "\n".join(f"{i}. [{'done' if s['id'] in finished else 'todo'}] {s['description']}"
                     for i, s in enumerate(state.get("plan", []), 1))


def _history(state: dict) -> str:
    lines = [f"- {d['step_id']}: {d['digest']}" for d in state.get("done", [])]
    lines += [f"- lesson from {x['step_id']}: tried \"{x['attempt']}\", failed with {x['failure']}"
              for x in state.get("lessons", [])]
    return "\n".join(lines) or "(nothing ran)"


def _waiting(state: dict) -> list[dict]:
    """Queued calls the user has not yet approved (nothing has run them)."""
    ran = set(state.get("executed", []))
    return [e for e in state.get("escalations", []) if e["id"] not in ran]


def _run_granted(waku, state: dict) -> dict:
    """Run the calls the user just approved, once, exactly as they were shown.

    Called only by the nodes that follow an Approval gate. A queued call is
    always followed by a Revision gate (the Step set `replan_requested`), so
    anything still waiting when execute_step or synthesize starts was approved.
    The harness runs it, not the model, so a retry cannot repeat the Step's
    other side effects.
    """
    done, executed = [], []
    for esc in _waiting(state):
        out = waku.tools.execute(esc["tool"], esc["args"])
        done.append({"step_id": esc["id"], "digest": f"approved call {esc['tool']}: {out}"})
        executed.append(esc["id"])
    return {"done": done, "executed": executed}


def _bind(run: _Run) -> object:
    """The compiled graph, its four callables closed over this run."""
    waku = run.waku

    def plan_fn(state: dict) -> dict:
        resp = waku.client.messages.create(
            model=waku.settings.model, max_tokens=800,
            messages=[{"role": "user", "content": PLAN_PROMPT.format(goal=state["goal"])}])
        return {"plan": _steps_from(_text(resp))}

    def execute_fn(state: dict) -> dict:
        step = next_step(state)
        if step is None:
            return {}
        history = _history(state)
        prompt = STEP_PROMPT.format(
            goal=state["goal"], plan=_plan_lines(state), step=step["description"],
            history=f"\nWhat has happened so far:\n{history}\n" if state.get("done") else "")
        granted = _run_granted(waku, state)
        scoped = _ScopedTools(waku.tools, step["id"], len(state.get("escalations", [])) + 1)
        try:
            result = run_loop(
                client=waku.client, model=waku.settings.model,
                system="You are Waku, carrying out one step of an approved plan.",
                messages=[{"role": "user", "content": prompt}], tools=scoped,
                max_iterations=waku.settings.max_iterations,
                max_tokens=waku.settings.max_tokens,
                observer=lambda k, ev: run.notify(k, {**ev, "node": "execute_step"}))
        except Exception as exc:
            # A failed Step is finished (it has a `done` entry) and leaves a
            # Lesson; the run goes to `replan` instead of taking the dashboard down.
            failure = error_text(exc)
            return {"done": [*granted["done"], {"step_id": step["id"],
                                                "digest": f"FAILED: {failure}"}],
                    "executed": granted["executed"],
                    "escalations": scoped.queued,
                    "lessons": [{"step_id": step["id"], "attempt": step["description"],
                                 "failure": failure}],
                    "replan_requested": True}
        return {"done": [*granted["done"], {"step_id": step["id"], "digest": result.reply}],
                "executed": granted["executed"], "escalations": scoped.queued,
                "replan_requested": bool(scoped.queued)}

    def replan_fn(state: dict) -> dict:
        waiting = _waiting(state)
        # Phase 1 ticket 02: the Revision keeps the remaining Steps and records
        # why. Producing a different tail with a model call is ticket 04.
        last = state["done"][-1]["step_id"] if state.get("done") else ""
        return {"revisions": [{"after_step": last, "change": "remaining Steps unchanged",
                               "reason": (f"{waiting[0]['tool']} needs your approval" if waiting
                                          else state["lessons"][-1]["failure"]
                                          if state.get("lessons") else "new work was discovered")}]}

    def synthesize_fn(state: dict) -> dict:
        granted = _run_granted(waku, state)
        state = {**state, "done": [*state.get("done", []), *granted["done"]]}
        # No tools parameter: a model with no tool schemas cannot act while summarising.
        resp = waku.client.messages.create(
            model=waku.settings.model, max_tokens=1000,
            messages=[{"role": "user", "content": SYNTH_PROMPT.format(
                goal=state["goal"], history=_history(state))}])
        return {"digest": _text(resp), "done": granted["done"], "executed": granted["executed"]}

    return build_plan_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                            replan_fn=replan_fn, synthesize_fn=synthesize_fn)


def _pending(state: dict) -> str:
    """What the user is being asked to approve."""
    lines = ["**Plan**", "", _plan_lines(state), "",
             "Steps may use on their own: " + ", ".join(sorted(ALLOWED_TOOLS)) + "."]
    waiting = _waiting(state)
    if waiting:
        lines += ["", "**Needs your approval** (approving runs these exactly as shown)"]
        lines += [f"- {e['step_id']} wants `{e['tool']}` with {json.dumps(e['args'])}"
                  for e in waiting]
    if state.get("lessons"):
        lines += ["", "**What went wrong**"]
        lines += [f"- {x['attempt']}: {x['failure']}" for x in state["lessons"]]
    lines += ["", "Type `/plan approve` to run it, or `/plan stop` to drop it."]
    return "\n".join(lines)


def _drive(run: _Run, first_input) -> dict:
    """Stream the graph and re-emit LangGraph's updates in Waku's vocabulary."""
    global _paused
    notify = run.notify
    cfg = {"configurable": {"thread_id": run.thread_id}}
    seen: dict[str, int] = {}
    t0 = last = time.perf_counter()
    notify("graph_start", {"workflow": "plan", "nodes": list(run.app.get_graph().nodes)})
    try:
        for chunk in run.app.stream(first_input, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node.startswith("__"):          # the interrupt marker, not a node
                    continue
                seen[node] = seen.get(node, 0) + 1
                now = time.perf_counter()
                notify("node_start", {"workflow": "plan", "node": node, "visit": seen[node]})
                notify("node_end", {"workflow": "plan", "node": node,
                                    "ms": int((now - last) * 1000),
                                    "keys": list((update or {}).keys()), "error": None})
                last = now
    except Exception as exc:
        _paused = None
        notify("graph_end", {"workflow": "plan", "ms": int((time.perf_counter() - t0) * 1000),
                             "steps": sum(seen.values()), "path": list(seen),
                             "error": error_text(exc)})
        return {"digest": f"The plan run failed: {error_text(exc)}"}
    snap = run.app.get_state(cfg)
    notify("graph_end", {"workflow": "plan", "ms": int((time.perf_counter() - t0) * 1000),
                         "steps": sum(seen.values()), "path": list(seen), "error": None})
    if snap.next:                                   # stopped at an Approval gate
        _paused = run
        return {"digest": _pending(snap.values)}
    _paused = None
    return {"digest": snap.values.get("digest") or "(the plan produced no text)"}


def run_plan(observer=None, message: str = "", waku=None) -> dict:
    """Plan a goal and run it, with the user approving first.

    Returns {"digest": reply} like the other workflows. While a run is paused
    the only valid messages are exactly `approve` and `stop`. The agent lock is
    held for the whole call, resume included, so a Step's tools never race a
    chat message; a caller that passes its own `waku` (a test) owns that.
    """
    if waku is None:
        with agent_lock:
            return _run_plan(observer, message, get_agent())
    return _run_plan(observer, message, waku)


def _run_plan(observer, message: str, waku) -> dict:
    global _paused
    text = (message or "").strip()

    if _paused is not None:
        if text not in ("approve", "stop"):
            pending = _pending(_paused.app.get_state(
                {"configurable": {"thread_id": _paused.thread_id}}).values)
            return {"digest": f"A plan is waiting for you.\n\n{pending}\n\n"
                              "Reply exactly `/plan approve` or `/plan stop`."}
        run, _paused = _paused, None
        if text == "stop":
            return {"digest": "Plan stopped. Nothing further was run."}
        run.observer = observer
        return _drive(run, Command(resume="approve"))

    if not text:
        return {"digest": USAGE}
    run = _Run(app=None, thread_id=uuid.uuid4().hex, waku=waku, observer=observer)
    run.app = _bind(run)
    return _drive(run, {"goal": text})
