# 07 — bridge: one step is one Waku loop turn

Status: done
Blocked by: 02, 03, 04, 05
Plan task: Task 7 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

The join. `run_loop` is a parameter, not an import, so tests substitute a fake
and never touch a network. `run_loop` itself is not modified.

The rollup filter here is the one thing in this ticket that is easy to get
wrong and invisible when wrong: `done` is append-only, so a rollup cannot
remove anything from it. If the prompt builder does not skip steps a rollup
covers, the rollup makes the prompt LONGER and every measurement in ticket 10
still passes.

## Done when

- Artifacts parse from the marker line, kept verbatim; a malformed pair is skipped, not fatal
- The prompt carries goal, plan, rollups, recent digests with artifacts, and lessons, and never a transcript
- `_history` omits any `done` entry already inside a rollup, proven by a test asserting the rolled-up digest is GONE and an unrolled one stays
- The usage meter turns the loop's real `llm` event shape into dollars and ignores other events
- A step writes its full output to disk and merges back a short digest
- A step that raises records a Lesson and asks for a replan, producing no result
- `guarded_registry` dedupes side-effecting tools per step, passes read-only tools through, and leaves the schemas the model sees unchanged
- 16 tests pass across bridge and idempotency, and the read-only pass-through is shown to fail with an empty READ_ONLY set
