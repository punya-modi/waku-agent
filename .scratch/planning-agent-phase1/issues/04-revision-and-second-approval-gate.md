# 04: Revision and the second approval gate

**What to build:** full handling of the case where a Step fails or discovers new work mid-run. The `replan` node (already routed to by ticket 01's topology, and already reachable via ticket 02's Lesson-on-failure handling) produces a real Revision — a replacement for the remaining, not-yet-run Steps, preserving every already-finished Step untouched — via a real model call, then pauses at the second Approval gate (the same `interrupt()` mechanism as the first gate, just inside `replan` instead of `plan`). `/plan approve` at that point resumes with the revised Plan; `/plan stop` abandons it, same as at the first gate.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] A Step engineered to fail (or to report discovered work) causes the run to produce a real Revision via a model call, rather than just recording a Lesson and stopping
- [ ] The revised Plan is shown to the user and the run pauses, exactly as it does after the initial `plan` — same `/plan approve`/`/plan stop` contract from ticket 02 applies
- [ ] Steps already finished before the Revision are never re-run or altered by it
- [ ] Approving at this second gate resumes execution using the revised remaining Plan, not the original one
- [ ] Manual check: a goal engineered to hit a conflict or a deliberately-failing Step shows the correct revised Plan before continuing
- [ ] Deterministic eval covers the Revision path end to end via the Seam B pattern (`ScriptedClient` + fake `ToolRegistry`), asserting the revised Plan content and that already-finished Steps are preserved
