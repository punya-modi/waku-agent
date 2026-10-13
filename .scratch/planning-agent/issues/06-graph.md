# 06 — graph: the topology and the code routers

Status: ready-for-agent
Blocked by: 02
Plan task: Task 6 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

Four injected node callables, two router functions. No node calls a model, so
this ticket runs at zero spend. Prior art: `build_triage_graph` in
`waku/graph/workflows/triage.py`.

## Done when

- An empty plan routes straight to synthesize
- A budget of zero or less halts before any step runs
- A replan wins over remaining work, but NOT over the replan cap
- An exhausted budget stops the cycle with work left
- The replan edge is conditional through `after_plan`, so a replan cannot buy a step past the budget
- The cycle runs every step once and stops
- No LLM call decides an edge
- 14 tests pass, and the budget guard is shown to fail when `after_plan` returns "step" unconditionally
