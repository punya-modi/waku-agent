# 11 — the findings and the graduation verdict

Status: done
Blocked by: 10
Plan task: Task 11 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

The topic ends in a decision, not a maybe.

## Done when

- The README's "What we found" carries the REAL numbers from ticket 10, not the plan's estimates
- It states that rollups slow growth rather than flattening it, with the measured ratio
- It records the two findings worth more than the verdict: resume re-runs the failed node, and a dedup table must record failures as failures
- "Graduation" states whether `waku/graph/` should take LangGraph or grow its own checkpointer, and at which rung of the footprint ladder
- The `create_event` skipped-sync hole is recorded as a Waku bug in its own right, worth an issue on `upstream` whatever happens to this topic
- One line is added to `docs/status.md`, with `**Last updated:**` current and the file still under 120 lines
- `pytest lab/planning-agent/tests`, `pytest evals/deterministic` and `make lint` all pass
