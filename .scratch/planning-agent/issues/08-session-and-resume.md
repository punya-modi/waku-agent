# 08 — session and the resume checks

Status: ready-for-agent
Blocked by: 05, 06
Plan task: Task 8 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

The headline capability and the reason the topic exists.

## Done when

- A finished task needs no resume
- A crashed task resumes on a fresh connection and does not redo completed work
- The paired test shows the dedup table is what keeps the world consistent
- The checkpoint does not grow with the number of finished steps
- A `create_event` replay is shown to report "already exists (not duplicated)", documenting why the planner keys its own calls
- ONE test really kills an interpreter: a child process commits a side effect, calls `os._exit(1)` before any checkpoint, and a second child resumes and asserts the external file holds exactly one line
- 6 tests pass
