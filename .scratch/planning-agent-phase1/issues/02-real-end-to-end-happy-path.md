# 02: Real end-to-end happy path on the dashboard

**What to build:** the binder (`waku/ops/plan.py`, exposing `run_plan(observer, message)`) that makes `/plan` a real, usable dashboard command, picked up automatically by `commands.discover()` the same way `/gather` and `/triage` are. `/plan <goal>` makes a real model call to decompose the goal into a Plan and shows it, paused at the first Approval gate from ticket 01. `/plan approve` resumes the paused run and executes every Step for real, one real turn of Waku's agent loop (`run_loop` + the real `ToolRegistry`) at a time — the same wiring move `waku/graph/nodes.py:agent_node` already makes, rewritten fresh (never imported from `lab/planning-agent/`) — until the Plan finishes and a real digest comes back. A Step that raises is caught, turned into a Lesson, and sets `replan_requested` rather than crashing the run (full replan/second-gate behavior is ticket 04; here it's enough that a failure doesn't crash).

Also covers the message-parsing contract: while nothing is paused, `message` is a new goal. While a thread is paused, `message` must be exactly `approve` or `stop` (a closed set); anything else replies with what's pending and the two valid replies, and leaves the paused state untouched. `stop` discards the paused slot without resuming. The currently-paused thread id lives in one module-level slot in `waku/ops/plan.py` — this is one dashboard, one user, one run at a time, no database.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Typing `/plan <goal>` in the dashboard chat produces a real, multi-step Plan (via a real model call) and the reply shows it along with what to type next
- [ ] Typing `/plan approve` after that runs every Step for real against real tools, and the final reply is a real digest of what happened
- [ ] A Step that raises an exception does not crash the run — it's recorded as a Lesson and the run routes toward `replan` instead of propagating the exception
- [ ] While a thread is paused, `/plan` with anything other than exactly `approve` or `stop` leaves the paused state untouched and explains what's pending
- [ ] `/plan stop` while paused discards the paused state without running anything further
- [ ] `/plan <goal>` while nothing is paused always starts a fresh run, same as any other slash command
- [ ] `run_plan`'s return shape matches what `waku/ops/dashboard.py:_run_command` already expects (a `digest` key, etc.) so no dashboard changes are needed to display the reply
- [ ] Manual acceptance check: "Look at my calendar this week and schedule a 30-minute focus block each weekday morning that doesn't already conflict, then leave me a note summarizing what you scheduled" — using only already-registered tools (`list_events`/`local_events`, `create_event`, `save_note`) — runs to completion through the dashboard chat
- [ ] Deterministic eval drives `run_plan` end to end through a `ScriptedClient` (standing in for both the planning call and each Step's `run_loop` call) plus the existing fake `ToolRegistry` test helpers — prior art: `evals/deterministic/test_triage_workflow.py`
