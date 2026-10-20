# LangGraph for the planning agent's Phase 1, ahead of when it's strictly needed

Phase 1 of the planning agent needs two pause points (after the initial Plan, and after any Revision) where it waits for the user to approve before continuing. That behavior alone doesn't require LangGraph: it maps onto two separate dashboard-triggered runs with state held in server memory between them, the same way `waku/ops/dashboard.py` already caches the Notion episode store — no checkpointer, no new dependency, no deviation from hard rule 3 (no new default dependency).

**Decision:** build Phase 1 on LangGraph anyway, deliberately, to get real practice with its `interrupt()` pause/resume idiom on a genuine use case — not because the approval gate needs it. This is a conscious trade of "more dependency than Phase 1 technically requires" for "hands-on fluency before Phase 2 needs it for real."

**Consequence worth knowing:** LangGraph's `interrupt()` requires a checkpointer bound at graph-compile time to work at all, even for an in-process pause with no crash-resume promised. So Phase 1 is not "LangGraph without the checkpointer" — it stands up a checkpointer (at minimum an in-memory one) from the start, collapsing some of what was scoped as Phase 2 work into Phase 1. Phase 2's job narrows to: swap the in-memory saver for a persistent one, and add the dedup/idempotency table for safe replay (the lab topic's `idempotency.py` finding, which is framework-agnostic either way).

Must still ship behind an optional extra (e.g. `[planning]`), per hard rule 3 and conventions §7 — this decision is "use LangGraph," not "make it a default dependency."
