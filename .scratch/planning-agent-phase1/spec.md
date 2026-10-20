Status: ready-for-agent

# Planning agent — Phase 1: runnable and watchable on the dashboard

## Problem Statement

A LangGraph-based planning agent was built as a `lab/` topic (`lab/planning-agent/`), but it only ever ran against stub nodes and a hand-written fake plan. Nobody has ever typed a real goal into Waku's dashboard, watched it get decomposed into real steps, and watched those steps actually run with real tools. The capability the user actually wants — a planning agent they can reach from the dashboard, give a real multi-step goal, watch think, and approve before it acts — doesn't exist yet.

## Solution

A `/plan` command, reachable from the dashboard chat exactly like `/gather` or `/triage` today. `/plan <goal>` decomposes the goal into a multi-step Plan using a real model call, shows that Plan on the dashboard with a live trace, and waits. `/plan approve` runs every Step for real, one real turn of Waku's agent loop at a time, against real tools, until the Plan is finished or a Step triggers a Revision — at which point it shows the revised remaining Plan and waits again. `/plan stop` abandons a paused run. Built on LangGraph, deliberately (see ADR-0001), using its `interrupt()`/checkpoint mechanism to implement the pause — not because Phase 1 strictly needs crash-survival, but because this is also deliberate hands-on practice with LangGraph's pause/resume idiom ahead of Phase 2, where that mechanism becomes load-bearing.

This is Phase 1 only. Crash-resume, replay-safety, prompt-growth management for long plans, and observability tooling are explicitly out of scope (see below) and belong to Phase 2.

## User Stories

1. As the dashboard user, I want to type `/plan <goal>` in the chat box, so that I can ask the planning agent to decompose a real goal without leaving the dashboard.
2. As the dashboard user, I want to see the full numbered Plan before any Step runs, so that nothing happens to my calendar, notes, or messages without my say.
3. As the dashboard user, I want the chat to tell me exactly what to type to continue (`/plan approve`) or cancel (`/plan stop`), so that I'm never left guessing what state the run is in.
4. As the dashboard user, I want `/plan approve` to run every Step for real, using the same tools and loop the rest of Waku uses, so that the planning agent's work is genuine, not simulated.
5. As the dashboard user, I want to watch each Step start and finish live on the dashboard's existing trace view, so that I can see the planning agent think, the same way I can for `/gather` and `/triage` today.
6. As the dashboard user, if a Step fails or discovers new work, I want the planning agent to revise the remaining Plan and show me the new version before continuing, so that I can catch a bad direction before more Steps run on top of it.
7. As the dashboard user, I want `/plan approve`/`/plan stop` to only be accepted while a run is actually paused, and anything else typed while paused to explain what's pending rather than silently misfire, so that a typo can't accidentally start an unrelated new goal or get swallowed.
8. As the dashboard user, I want `/plan <new goal>` to just work as a fresh run when nothing is paused, so that the command behaves like any other slash command when there's nothing to resume.
9. As the maintainer, I want the planning agent to follow the exact same `workflows/<name>.py` + `ops/<name>.py:run_<name>` convention as `triage`/`gather`, so that `/plan` is discoverable via the existing `commands.discover()` mechanism with no new registration code.
10. As the maintainer, I want the pure graph topology (`waku/graph/workflows/plan.py`) to take injected node callables with no client or filesystem bound inside, so that its routing and interrupt behavior is testable at zero cost, same as `triage.py`/`gather.py`.
11. As the maintainer, I want the LangGraph dependency behind an optional extra (`[planning]`), so that installing Waku's default package still means stdlib + the two model clients (hard rule 3).
12. As the maintainer, I want a Step that raises an exception to be caught and turned into a Lesson plus a Revision request, rather than crashing the whole run, so that one bad tool call doesn't take down the dashboard.
13. As the maintainer, I want the LangGraph stream's own event vocabulary translated into Waku's `node_start`/`node_end`/etc. vocabulary before it reaches the dashboard's observer, so that the existing live-trace rendering and topology chart work unmodified.
14. As a future contributor working on Phase 2, I want this Phase 1 implementation to not have built an idempotency/dedup table or a rollup/offload mechanism it doesn't need yet, so that Phase 1 stays legible and Phase 2 can add exactly the mechanism its own requirement (crash-safe replay, long-plan prompt growth) calls for.
15. As the maintainer, I want every behavior this spec introduces covered by a deterministic eval that runs offline with no API key, so that `make gate` and CI catch a regression without needing a provider key (hard rule 4).

## Implementation Decisions

- **New optional dependency**: `pyproject.toml` gains `[project.optional-dependencies] planning = ["langgraph"]`. Nothing else — Phase 1 only uses LangGraph's built-in `MemorySaver`, not `langgraph-checkpoint-sqlite`.
- **`waku/graph/workflows/plan.py`** (pure, Seam A): builds a LangGraph `StateGraph` over a `PlanState` (goal, plan, done, lessons, revisions, replan_requested — modeled on the vocabulary in `CONTEXT.md`: Goal, Plan, Step, Revision). Takes `plan_fn`, `execute_fn`, `replan_fn`, `synthesize_fn` as injected callables — no client or filesystem reference inside this module, matching `triage.py`'s `build_triage_graph` shape. Topology: `plan → execute_step → (loop while Plan has unfinished Steps) → replan (only if a Step requested it) → synthesize`. The Approval gate from `CONTEXT.md` is LangGraph's `interrupt()`, and it only fires at two points, never after an individual Step: right after `plan` and right after `replan`. Each gate is its own node (`approve_plan`, `approve_revision`), not a call inside `plan_fn`/`replan_fn`: LangGraph re-runs a node from the top on resume, so an `interrupt()` inside `plan` would call the model again and could execute a different Plan from the one the user approved. Topology: `plan → approve_plan → execute_step → (loop while Plan has unfinished Steps) → replan → approve_revision → execute_step`, ending in `synthesize`. These gate nodes also appear in the event stream the binder translates.
- **`waku/ops/plan.py`** (binder, Seam B): exposes `run_plan(observer, message)`, which `commands.discover()` picks up automatically as `/plan`, following the existing `workflows/<name>.py` + `ops/<name>.py:run_<name>` convention. Holds one module-level slot for the currently-paused thread id (`None` when nothing is paused) — a single global is sufficient because this is one dashboard, one user, one run at a time; no database. Compiles the LangGraph app once with `MemorySaver()` and keeps the compiled app + saver alive at module scope for as long as a thread stays paused (the saver instance must survive between the two separate `run_plan` calls, or the resume has nothing to resume from). `execute_fn` is built from Waku's real `run_loop` plus the real `ToolRegistry`, the same wiring move `waku/graph/nodes.py:agent_node` already makes — this is rewritten fresh, not imported from `lab/planning-agent/bridge.py`, per the rule that nothing under `waku/` may import from `lab/`.
- **Message parsing in `run_plan`**: if no thread is currently paused, `message` is treated as a new goal and a fresh LangGraph thread starts. If a thread is paused, `message` must be exactly `approve` or `stop` — a closed set, not "anything that doesn't look like a goal" — per conventions §4's guard-as-closed-set rule. `approve` resumes via `app.invoke(Command(resume="approve"), config={"configurable": {"thread_id": thread_id}})`; `stop` discards the paused slot without resuming. Anything else while paused returns a message describing what's pending and the two valid replies, without touching the paused state.
- **Step failure handling**: inside `execute_fn`, a Step that raises is caught (not allowed to propagate), turned into a Lesson, and sets `replan_requested` — landing the run at the next `interrupt()` gate via the `replan` node, mirroring the exception handling already proven in `lab/planning-agent/bridge.py:make_execute_step`.
- **Event translation**: `run_plan` consumes LangGraph's own `.stream(..., stream_mode="updates")` events and re-emits them through `observer(kind, event)` using Waku's existing vocabulary (`node_start`, `node_end`, etc., as `waku/graph/engine.py` already defines it), so the dashboard's SSE handling and topology chart (`waku/ops/dashboard.py:graph_stream`) need no changes to render a `/plan` run.
- **No new dashboard route.** The approval gate is answered through the existing chat/slash-command path (`/plan approve`, `/plan stop`), which needs no entry in `evals/deterministic/test_dashboard_routes.py` and no `hosted/core/policy.py` decision, since no new route is added.
- **Docs**: `CONTEXT.md` and `docs/adr/0001-langgraph-for-planning-agent-phase-1.md` are already written and should be treated as settled vocabulary/rationale for this spec, not re-derived.

## Testing Decisions

A good test here asserts externally observable behavior — the Plan/Revision/digest content a caller gets back, or the sequence of `observer` events a caller sees — never internal LangGraph call shapes.

- **Seam A** (`evals/deterministic/`, new file, e.g. `test_plan_workflow.py`): drives `build_plan_graph()` directly with harmless stub `plan_fn`/`execute_fn`/`replan_fn`/`synthesize_fn` and LangGraph's own `.invoke()`/`Command(resume=...)`. No model, no network, no tool registry. Prior art: `evals/deterministic/test_gather_workflow.py`'s `build_gather_graph(**overrides)` pattern. Must assert: `interrupt()` fires after `plan` and after `replan`, never after an individual Step; approving resumes rather than restarts; a Step requesting a Revision routes to `replan`, not straight back to `execute_step`.
- **Seam B** (`evals/deterministic/`, new file, e.g. `test_plan_workflow_binder.py` or similar): drives `run_plan` end to end through a `ScriptedClient` standing in for the model (both the planning call and each executed Step's `run_loop` call) plus the repo's existing fake `ToolRegistry` test helpers. Prior art: `evals/deterministic/test_triage_workflow.py`'s full-app-with-`ScriptedClient` pattern. Must assert: `/plan <goal>` → paused-with-plan → `/plan approve` → steps actually execute via the scripted tool calls → finished digest; `/plan` while paused with anything other than `approve`/`stop` leaves the paused state untouched and explains what's pending; the `{"digest": ...}` return shape matches what `waku/ops/dashboard.py:_run_command` already expects; Waku-vocabulary `node_start`/`node_end` events reach the `observer` during a real run (covering the event-translation layer as a side effect of Seam B rather than as its own isolated unit).
- Every behavior named in the User Stories above gets covered by one of these two seams — no behavior change ships without a red-then-green deterministic eval (hard rule 4).

## Out of Scope

Explicitly deferred to Phase 2, not part of this spec:

- Crash-resume / surviving a process restart (Phase 1's `MemorySaver` is in-memory only; losing the dashboard process loses any paused run, and that's accepted).
- The idempotency/dedup table for safe replay of a resumed run (only matters once crash-resume exists — a Phase 1 Step only ever runs once).
- Rollups / disk-offload for prompt growth on long plans (only matters once plans run past the length Phase 1's test goal produces).
- Any deep observability/tracing tooling beyond the dashboard's existing live-trace view.
- Per-Step approval (approving before every individual Step rather than once-and-on-revision — considered and rejected for Phase 1; see `CONTEXT.md`).
- Reject-and-redo or hand-editing a Step at an approval gate (approve/stop only).
- A dedicated approve/stop dashboard UI (button, card) — Phase 1 rides entirely on the existing chat/slash-command surface.
- Multiple concurrent in-flight plans, or multi-user support — the module-level single-paused-thread slot assumes exactly one.
- `langgraph-checkpoint-sqlite` or any persistent checkpointer backend.

## Further Notes

- First manual acceptance scenario (not a test, a sanity check once this is implemented): "Look at my calendar this week and schedule a 30-minute focus block each weekday morning that doesn't already conflict, then leave me a note summarizing what you scheduled." Decomposes into a real multi-step Plan using only already-registered tools (`list_events`/`local_events`, `create_event`, `save_note`) — deliberately avoids `send_message` so no real person is pinged while this is being exercised for the first time.
- `lab/planning-agent/`'s own graduation verdict recommended *against* taking the LangGraph dependency, porting only its checkpointer idea into `waku/graph/engine.py` instead. This spec deliberately goes the other way for Phase 1 — see ADR-0001 for why, and don't re-litigate that trade-off without re-reading it first.
- The `create_event` dedup-ordering bug the lab topic found (a replay can be told "already exists" before the ics/Apple/Google syncs run) is a real `waku/` bug independent of this spec. It isn't fixed here (Phase 1 never replays a Step), but it becomes directly relevant the moment Phase 2 adds crash-resume, and is worth filing as its own issue on `upstream` before then.
