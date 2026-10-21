"""DETERMINISTIC EVAL — `/plan`: the binder that makes the plan graph runnable.

Drives `run_plan` end to end the way the dashboard does (observer + message),
with a ScriptedClient standing in for the model and the real ToolRegistry over
a throwaway home. SCRIPTED-RESPONSE ORDER, read before adding cases: the
planning call comes first, then each Step's run_loop responses in Plan order,
then (after the last Step) the synthesis call.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph", reason="the planning agent is the [planning] extra")

from evals.helpers import ScriptedClient, make_waku, response, text_block, tool_block  # noqa: E402
from waku.ops import commands  # noqa: E402
from waku.ops import plan as plan_binder
from waku.ops.plan import run_plan  # noqa: E402


@pytest.fixture(autouse=True)
def _no_leftover_pause():
    plan_binder._paused = None
    yield
    plan_binder._paused = None


def planned(*steps: str):
    return response([text_block(json.dumps({"steps": list(steps)}))])


def run(app, message, events=None):
    obs = (lambda k, ev: events.append((k, ev))) if events is not None else None
    return run_plan(observer=obs, message=message, waku=app)


def two_step_app(tmp_path, *extra):
    script = [
        planned("Book the swim", "Note what was booked"),
        response([tool_block("create_event", {"title": "Swim", "start": "2026-08-01 09:00",
                                              "end": "2026-08-01 10:00"})], "tool_use"),
        response([text_block("Swim booked for Saturday.")]),
        response([text_block("Noted.")]),
        response([text_block("All done: swim booked and noted.")]),
        *extra,
    ]
    return make_waku(tmp_path / "home", client=ScriptedClient(script))


def test_plan_is_discovered_as_a_slash_command():
    assert commands.discover()["plan"] == "waku.ops.plan:run_plan"


def test_a_goal_shows_the_plan_and_pauses_before_any_step_runs(tmp_path):
    app = two_step_app(tmp_path)
    out = run(app, "book a swim and note it")
    assert "Book the swim" in out["digest"] and "Note what was booked" in out["digest"]
    assert "/plan approve" in out["digest"] and "/plan stop" in out["digest"]
    assert plan_binder._paused is not None
    assert not (app.settings.home / "calendar.ics").exists(), "no Step ran before approval"


def test_approve_runs_every_step_for_real_and_returns_the_digest(tmp_path):
    app = two_step_app(tmp_path)
    run(app, "book a swim and note it")
    out = run(app, "approve")
    assert out["digest"] == "All done: swim booked and noted."
    assert "Swim" in (app.settings.home / "calendar.ics").read_text(encoding="utf-8")
    assert plan_binder._paused is None


def test_waku_vocabulary_events_reach_the_observer(tmp_path):
    app = two_step_app(tmp_path)
    events: list = []
    run(app, "book a swim and note it", events)
    run(app, "approve", events)
    kinds = [k for k, _ in events]
    assert "node_start" in kinds and "node_end" in kinds
    started = [ev["node"] for k, ev in events if k == "node_start"]
    assert started[:2] == ["plan", "approve_plan"]
    assert started.count("execute_step") == 2 and started[-1] == "synthesize"
    assert all(ev["workflow"] == "plan" for k, ev in events if k in ("node_start", "node_end"))
    assert "tool" in kinds, "the Step's loop events pass through to the observer"


def test_anything_but_approve_or_stop_leaves_the_pause_untouched(tmp_path):
    app = two_step_app(tmp_path)
    run(app, "book a swim and note it")
    held = plan_binder._paused
    out = run(app, "sure, go ahead")
    assert "approve" in out["digest"] and "stop" in out["digest"]
    assert "Book the swim" in out["digest"], "says what is pending"
    assert plan_binder._paused is held
    assert run(app, "approve")["digest"].startswith("All done")


def test_stop_discards_the_pause_without_running_anything(tmp_path):
    app = two_step_app(tmp_path)
    run(app, "book a swim and note it")
    out = run(app, "stop")
    assert plan_binder._paused is None
    assert "stopped" in out["digest"].lower()
    assert not (app.settings.home / "calendar.ics").exists()


def test_a_new_goal_when_nothing_is_paused_starts_a_fresh_run(tmp_path):
    app = two_step_app(tmp_path)
    run(app, "first goal")
    run(app, "stop")
    app.client._script[:0] = [planned("Only step")]
    out = run(app, "second goal")
    assert "Only step" in out["digest"] and "Book the swim" not in out["digest"]


def test_approve_with_nothing_paused_is_a_goal_not_a_resume(tmp_path):
    app = make_waku(tmp_path / "home", client=ScriptedClient([planned("Say hi")]))
    out = run(app, "approve")
    assert "Say hi" in out["digest"] and plan_binder._paused is not None


def test_a_step_that_raises_becomes_a_lesson_and_routes_to_replan(tmp_path, monkeypatch):
    app = two_step_app(tmp_path)
    run(app, "book a swim and note it")

    real = plan_binder.run_loop
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("calendar exploded")
        return real(**kwargs)

    monkeypatch.setattr(plan_binder, "run_loop", flaky)
    events: list = []
    out = run(app, "approve", events)   # must not raise
    started = [ev["node"] for k, ev in events if k == "node_start"]
    assert "replan" in started and "synthesize" not in started
    assert "calendar exploded" in out["digest"], "the pending revision shows the lesson"
    assert plan_binder._paused is not None, "paused at the second gate"
    assert "/plan approve" in out["digest"]


def test_a_planning_failure_answers_in_the_chat_and_leaves_nothing_paused(tmp_path):
    app = make_waku(tmp_path / "home",
                    client=ScriptedClient([response([text_block("no json, sorry")])]))
    out = run(app, "do the thing")
    assert plan_binder._paused is None
    assert "plan" in out["digest"].lower()


def escalating_app(tmp_path):
    from waku.tools.registry import Tool

    sent: list[dict] = []
    app = make_waku(tmp_path / "home", client=ScriptedClient([
        planned("Tell Sam the plan"),
        response([tool_block("send_message", {"to": "sam", "text": "hi"})], "tool_use"),
        response([text_block("Could not message Sam.")]),
        response([text_block("Asked for approval to message Sam.")]),
    ]))
    app.tools.register(Tool("send_message", "spy", {"type": "object"},
                            lambda **kw: sent.append(kw) or "sent"))
    return app, sent


def test_the_plan_reply_lists_what_a_step_may_do_alone(tmp_path):
    app = two_step_app(tmp_path)
    digest = run(app, "book a swim")["digest"]
    assert "create_event" in digest and "send_message" not in digest


def test_a_tool_outside_the_allowlist_is_refused_and_queued_for_the_next_gate(tmp_path):
    app, sent = escalating_app(tmp_path)
    run(app, "tell sam")
    out = run(app, "approve")
    assert sent == [], "the Step could not send it alone"
    assert "Needs your approval" in out["digest"] and "send_message" in out["digest"]
    assert '"to": "sam"' in out["digest"], "shows the exact arguments"
    assert plan_binder._paused is not None


def test_approving_the_escalation_runs_that_exact_call_once(tmp_path):
    app, sent = escalating_app(tmp_path)
    run(app, "tell sam")
    run(app, "approve")
    out = run(app, "approve")
    assert sent == [{"to": "sam", "text": "hi"}]
    assert out["digest"] == "Asked for approval to message Sam."
    assert plan_binder._paused is None


def test_stopping_at_the_escalation_gate_never_runs_the_call(tmp_path):
    app, sent = escalating_app(tmp_path)
    run(app, "tell sam")
    run(app, "approve")
    run(app, "stop")
    assert sent == [] and plan_binder._paused is None


def test_an_empty_plan_command_while_paused_shows_what_is_pending(tmp_path):
    app = two_step_app(tmp_path)
    run(app, "book a swim and note it")
    held = plan_binder._paused
    out = run(app, "")
    assert "Book the swim" in out["digest"] and plan_binder._paused is held


def test_the_default_path_holds_the_agent_lock_through_a_resume(tmp_path, monkeypatch):
    app = two_step_app(tmp_path)
    monkeypatch.setattr(plan_binder, "get_agent", lambda: app)
    seen = []
    real = plan_binder.run_loop
    monkeypatch.setattr(plan_binder, "run_loop",
                        lambda **kw: (seen.append(plan_binder.agent_lock.locked()), real(**kw))[1])
    run_plan(message="book a swim and note it")
    run_plan(message="approve")
    assert seen and all(seen), "every Step ran while the agent lock was held"


def test_run_plan_without_a_message_explains_usage():
    assert "/plan" in run_plan(observer=None, message="")["digest"]
