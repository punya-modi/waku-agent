# 05 — idempotency: so resume cannot repeat a side effect

Status: ready-for-agent
Blocked by: 01
Plan task: Task 5 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

This lands BEFORE the resume demonstration. Resuming re-runs the whole failed
node, so a step that commits a write and then raises commits it twice.

## Done when

- A tool not named read-only is treated as side-effecting, including one nobody has written yet
- The key changes with thread, step, tool and args, and ignores argument order
- Two tasks doing the same thing both act (thread id is in the key)
- A replayed successful call returns the recorded result without reaching the tool
- A failed call is recorded with status `error` and is NOT replayed; the retry reaches the tool
- A raised call is recorded as an error and re-raised
- `setup()` refuses a table that predates the `status` column rather than silently using it
- The paired graph test shows `['a','b','b','c']` without the table and `['a','b','c']` with it
- 10 tests pass
