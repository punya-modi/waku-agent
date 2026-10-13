# Planning agent

Status: done

The spec for this feature is repo documentation, not a scratch file, because it
is versioned with the code it describes:

- **Scope and spec:** `docs/superpowers/specs/2026-10-11-planning-agent-spec.md`
- **System design:** `docs/superpowers/specs/2026-10-11-planning-agent-design.md`
- **Task breakdown, with the code and tests for each task:** `docs/superpowers/plans/2026-10-11-planning-agent.md`

This directory holds only the work state: one ticket per task, with blocking
edges. Each ticket is a pointer plus the acceptance criteria that matter; the
detail lives in the plan task it names, which carries the failing test, the
implementation and the commit message.

## Working order

```
01 topic ─┬─> 02 state ─┬─> 03 digest ─┐
          │             ├─> 06 graph ──┼─> 08 session+resume  (also needs 05)
          ├─> 04 offload│             └─> 09 replan
          └─> 05 idempotency
                      02,03,04,05 ──> 07 bridge ──> 10 rollups+run ──> 11 findings
```

After 01, three tickets open at once: 02, 04 and 05. Ticket 07 is the join.

## Before starting any ticket

`pip install langgraph langgraph-checkpoint-sqlite` — neither is in the
default install, and nothing in `pyproject.toml` changes. Record the versions
you actually resolved in the topic README's `Verified against:` line.
