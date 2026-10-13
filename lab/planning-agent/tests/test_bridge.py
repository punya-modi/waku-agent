import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.bridge import (
    ARTIFACTS_MARKER, UsageMeter, make_execute_step, parse_artifacts, step_prompt,
)
from lab.planning_agent.offload import Scratch
from lab.planning_agent.state import new_state


class _FakeResult:
    def __init__(self, reply, tool_calls=None):
        self.reply = reply
        self.tool_calls = tool_calls or []
        self.iterations = 1
        self.limit_reached = False


def _fake_run_loop(reply, usage=(1000, 500)):
    """Stands in for waku.loop.agent.run_loop, including its observer calls."""
    def run_loop(*, client, model, system, messages, tools, max_iterations,
                 max_tokens, observer=None, trim=None, **kwargs):
        if observer:
            observer("llm", {"iteration": 1, "usage": {"in": usage[0], "out": usage[1]}})
        return _FakeResult(reply)
    return run_loop


def test_artifacts_are_parsed_from_the_marker_line():
    reply = f"Booked it.\n{ARTIFACTS_MARKER} event_id=abc123; url=https://x.test/1"
    assert parse_artifacts(reply) == {"event_id": "abc123", "url": "https://x.test/1"}


def test_a_reply_with_no_marker_has_no_artifacts():
    assert parse_artifacts("just prose") == {}


def test_an_identifier_is_kept_verbatim_rather_than_summarised():
    """The reason artifacts exist: a digest may paraphrase an id, and a later
    step needs it exactly."""
    reply = f"done\n{ARTIFACTS_MARKER} id=EV-2026-10-11-0001"
    assert parse_artifacts(reply)["id"] == "EV-2026-10-11-0001"


def test_a_malformed_artifact_pair_is_skipped_not_fatal():
    reply = f"x\n{ARTIFACTS_MARKER} good=1; nonsense; alsogood=2"
    assert parse_artifacts(reply) == {"good": "1", "alsogood": "2"}


def test_the_prompt_carries_the_goal_the_plan_and_earlier_digests():
    state = new_state("throw a party", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "find a venue", "status": "pending"},
                     {"id": "b", "description": "book it", "status": "pending"}]
    state["done"] = [{"step_id": "a", "digest": "three venues found",
                      "artifacts": {"shortlist": "v1,v2"}, "output_path": "p"}]
    prompt = step_prompt(state, state["plan"][1])

    assert "throw a party" in prompt
    assert "find a venue" in prompt and "book it" in prompt
    assert "three venues found" in prompt
    assert "shortlist=v1,v2" in prompt
    assert "book it" in prompt.split("Your step now")[1]


def test_the_prompt_carries_lessons_so_a_retry_does_not_repeat_them():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "A", "status": "pending"}]
    state["lessons"] = [{"step_id": "a", "attempt": "called the venue",
                         "failure": "line was dead", "diagnosis": "use email"}]
    prompt = step_prompt(state, state["plan"][0])
    assert "line was dead" in prompt and "use email" in prompt


def test_the_prompt_carries_rollups_instead_of_the_digests_they_replaced():
    """The rollup has to REPLACE the digests, not join them. `done` is
    append-only, so nothing can leave it; this filter is the only place the
    collapse reaches the prompt. Without it a rollup makes the prompt longer."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "z", "description": "Z", "status": "pending"}]
    state["done"] = [
        {"step_id": "a", "digest": "venue A was free", "artifacts": {}, "output_path": ""},
        {"step_id": "b", "digest": "caterer B replied", "artifacts": {}, "output_path": ""},
        {"step_id": "c", "digest": "not yet rolled up", "artifacts": {}, "output_path": ""},
    ]
    state["rollups"] = [{"step_ids": ["a", "b"], "summary": "phase one settled the venue"}]
    prompt = step_prompt(state, state["plan"][0])

    assert "phase one settled the venue" in prompt
    assert "venue A was free" not in prompt, "a rolled-up digest must be gone"
    assert "caterer B replied" not in prompt
    assert "not yet rolled up" in prompt, "an unrolled digest stays"


def test_a_usage_meter_turns_token_counts_into_dollars():
    meter = UsageMeter(price_in=3.0, price_out=15.0)  # USD per million
    meter("llm", {"usage": {"in": 1_000_000, "out": 0}})
    assert meter.cost == pytest.approx(3.0)
    meter("llm", {"usage": {"in": 0, "out": 1_000_000}})
    assert meter.cost == pytest.approx(18.0)


def test_the_meter_ignores_events_that_are_not_llm_calls():
    meter = UsageMeter(price_in=3.0, price_out=15.0)
    meter("tool", {"tool": "x"})
    assert meter.cost == 0.0


def test_a_step_writes_its_full_output_and_merges_back_a_digest(tmp_path):
    state = new_state("g", budget_limit=5.0)
    state["plan"] = [{"id": "s1", "description": "do the thing", "status": "pending"}]
    body = "y" * 4000
    execute = make_execute_step(
        run_loop=_fake_run_loop(f"{body}\n{ARTIFACTS_MARKER} id=7"),
        client=object(), model="m", system="s", tools=object(),
        scratch=Scratch(tmp_path), thread_id="t", price_in=3.0, price_out=15.0)

    merged = execute(state)
    result = merged["done"][0]

    assert result["step_id"] == "s1"
    assert result["artifacts"] == {"id": "7"}
    assert len(result["digest"]) < 800, "the digest must be short"
    assert Scratch(tmp_path).read("s1").startswith("y" * 100), "the full output is on disk"
    assert merged["budget_spent"] > 0.0


def test_a_step_that_raises_records_a_lesson_and_asks_for_a_replan(tmp_path):
    state = new_state("g", budget_limit=5.0)
    state["plan"] = [{"id": "s1", "description": "do the thing", "status": "pending"}]

    def boom(**kwargs):
        raise RuntimeError("the provider refused")

    merged = make_execute_step(
        run_loop=boom, client=object(), model="m", system="s", tools=object(),
        scratch=Scratch(tmp_path), thread_id="t", price_in=3.0, price_out=15.0)(state)

    assert merged["replan_requested"] is True
    assert merged["lessons"][0]["step_id"] == "s1"
    assert "refused" in merged["lessons"][0]["failure"]
    assert merged["done"] == [], "a failed step produces no result"


def test_a_step_with_nothing_left_to_do_is_a_no_op(tmp_path):
    state = new_state("g", budget_limit=5.0)
    execute = make_execute_step(
        run_loop=_fake_run_loop("x"), client=object(), model="m", system="s",
        tools=object(), scratch=Scratch(tmp_path), thread_id="t",
        price_in=3.0, price_out=15.0)
    assert execute(state) == {}


import sqlite3

from waku.tools.registry import Tool, ToolRegistry

from lab.planning_agent.idempotency import DedupTable, guarded_registry


def _registry(calls):
    registry = ToolRegistry()

    def create_event(title: str = "", start: str = "") -> str:
        calls.append(("create_event", title))
        return f"created {title}"

    def list_events(day: str = "") -> str:
        calls.append(("list_events", day))
        return "two events"

    for name, fn in (("create_event", create_event), ("list_events", list_events)):
        registry.register(Tool(name=name, description=name,
                               input_schema={"type": "object", "properties": {}}, fn=fn))
    return registry


def _table():
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    return table


def test_a_side_effecting_tool_runs_once_per_step_however_often_it_is_called():
    calls: list[tuple] = []
    guarded = guarded_registry(_registry(calls), _table(), "t1", "s1")

    first = guarded.execute("create_event", {"title": "lunch", "start": "x"})
    second = guarded.execute("create_event", {"title": "lunch", "start": "x"})

    assert first == second == "created lunch"
    assert calls == [("create_event", "lunch")], "the tool must run once"


def test_a_read_only_tool_is_not_deduped():
    # Re-reading is free and must stay free: a step that lists events twice
    # should see the second answer, not the first one replayed.
    calls: list[tuple] = []
    guarded = guarded_registry(_registry(calls), _table(), "t1", "s1")
    guarded.execute("list_events", {"day": "monday"})
    guarded.execute("list_events", {"day": "monday"})
    assert calls == [("list_events", "monday"), ("list_events", "monday")]


def test_the_same_call_in_a_later_step_is_a_different_call():
    calls: list[tuple] = []
    table = _table()
    registry = _registry(calls)
    guarded_registry(registry, table, "t1", "s1").execute("create_event", {"title": "a"})
    guarded_registry(registry, table, "t1", "s2").execute("create_event", {"title": "a"})
    assert len(calls) == 2, "step 2 intends a new action, not a replay"


def test_the_schemas_the_model_sees_are_unchanged():
    guarded = guarded_registry(_registry([]), _table(), "t1", "s1")
    names = sorted(s["name"] for s in guarded.schemas())
    assert names == ["create_event", "list_events"]
