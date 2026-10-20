# 03: Live trace of a running plan

**What to build:** a translation layer inside `run_plan` that consumes LangGraph's own `.stream(..., stream_mode="updates")` events during a `/plan approve` run and re-emits them through `observer(kind, event)` using Waku's existing vocabulary (`node_start`, `node_end`, etc., as `waku/graph/engine.py` already defines it) — so the dashboard's existing live-trace rendering and topology chart (`waku/ops/dashboard.py:graph_stream`) light up step by step while a plan executes, exactly as they already do for `/gather` and `/triage`, with no changes needed to the dashboard itself.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Watching the dashboard during a `/plan approve` run shows each Step starting and finishing live, the same way `/gather`'s scans or `/triage`'s route show up today
- [ ] The events reaching `observer` use Waku's vocabulary (`node_start`/`node_end`/etc.), not LangGraph's native event shape
- [ ] Deterministic eval asserts the translated event sequence a caller sees during a scripted run (as a side effect of driving `run_plan`, per the Seam B pattern from ticket 02 — no new test infrastructure needed)
