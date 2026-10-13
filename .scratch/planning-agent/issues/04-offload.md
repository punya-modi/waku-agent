# 04 — offload: the scratch directory, read and searchable

Status: done
Blocked by: 01
Plan task: Task 4 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

Full outputs live on disk, not in the checkpoint: a checkpointer re-serialises
state on every step, so a transcript in state is rewritten at every later one.

## Done when

- What is written reads back whole, including a 10k body
- Search returns every matching line with its step id, case-insensitively
- Reading a step never written raises KeyError with an explicit message
- An unsafe step id is refused on both write and read, against a closed character set
- The guard is shown to fail when the pattern is loosened to `.+`
- 12 tests pass (the parametrised case counts six)
