# 03 — digest: short forms of a step and of a finished phase

Status: done
Blocked by: 02
Plan task: Task 3 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

Deterministic, no model, so tests pin exact strings and the whole thing is free.

## Done when

- A short output returns whole; whitespace collapses; an empty output gives an empty string
- A long output is cut and says how long it was
- A rollup names its step ids and joins their digests; a rollup of nothing is empty rather than an error
- `STEP_DIGEST_CHARS` is 600 and `ROLLUP_CHARS` is 800. Ticket 10 asserts the first of these, because the measured 2.2x saving depends on it
- 6 tests pass
