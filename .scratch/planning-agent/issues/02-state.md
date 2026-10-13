# 02 — state: the records, the reducers, next-step selection

Status: done
Blocked by: 01
Plan task: Task 2 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

The two reducers are the design: `plan` is replaced on every write, everything
else appends and is never rewritten.

## Done when

- `plan` replaced and `done` appended, proven through a real compiled graph
- `next_step` selects by id, never by position, and returns None on an empty plan
- `next_step` survives a revision that shortens the plan below `len(done)`
- `next_step` skips a step with `MAX_STEP_ATTEMPTS` lessons against it, and does not skip one with a single lesson
- `replans` exists and starts at 0; `strategy` does NOT exist
- `_path.py` installs the `lab.planning_agent` alias, loaded by file path, with `tests/conftest.py` as a thin loader over it
- 8 tests pass, and the reducer test is shown to fail when the appending annotation is removed
