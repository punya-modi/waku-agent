# 01: Pure plan graph with two approval gates

**What to build:** the LangGraph topology for the planning agent — `plan → execute_step (loop while the Plan has unfinished Steps) → replan (only if a Step requested it) → synthesize` — built entirely from injected callables (`plan_fn`, `execute_fn`, `replan_fn`, `synthesize_fn`), with no client or filesystem reference inside the module, matching the shape `triage.py`'s `build_triage_graph` already uses. The Approval gate (see `CONTEXT.md`) is LangGraph's `interrupt()`, and it must never fire after an individual Step. It sits in its own node right after `plan` (`approve_plan`) and right after `replan` (`approve_revision`), not inside them: LangGraph re-runs a node from the top on resume, so an `interrupt()` inside `plan` would call the model again and could execute a different Plan from the one the user approved. Also add the `[planning]` optional dependency (`langgraph` only, no `langgraph-checkpoint-sqlite`) to `pyproject.toml`, since this is the first ticket to import LangGraph at all.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `pyproject.toml` has a `[project.optional-dependencies] planning = ["langgraph"]` entry
- [ ] `build_plan_graph(plan_fn=..., execute_fn=..., replan_fn=..., synthesize_fn=...)` exists in `waku/graph/workflows/plan.py`, with no provider client or filesystem access referenced inside the module
- [ ] Driving the compiled graph with harmless stub callables and LangGraph's `.invoke()` shows `interrupt()` firing exactly once after `plan`, and again only if a Step sets `replan_requested` — never after a Step that didn't
- [ ] Resuming a paused run with `Command(resume="approve")` continues execution from where it paused (not from the start) and reaches `synthesize` once the Plan has no unfinished Steps
- [ ] A Step that sets `replan_requested` routes to `replan`, not straight back to `execute_step`
- [ ] Deterministic eval (e.g. `evals/deterministic/test_plan_workflow.py`) covers all of the above offline, no model, no network — prior art: `evals/deterministic/test_gather_workflow.py`
