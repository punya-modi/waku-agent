# 09 — the replanner, and lessons that get read

Status: ready-for-agent
Blocked by: 02
Plan task: Task 9 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

Roll back the work, keep the reflection. `done` is append-only, so this cannot
rewrite history even by mistake.

## Done when

- Finished work is preserved and only the remainder is restructured
- The request is cleared so the cycle can continue
- A Revision is appended saying what changed and why
- The proposer is given the goal, the unfinished steps, the finished results and the lessons
- A proposal of nothing leaves no work and ends the task
- A replan that shortens the plan below `len(done)` still selects sensibly
- A proposer that raises leaves the plan alone, clears the request, and records the failure
- Every replan increments `replans`, including one that raises
- 9 tests pass, and leaving the request set is shown to route back into the replanner for ever
