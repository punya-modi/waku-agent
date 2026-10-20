# Waku Agent

A local-first personal assistant built around four pillars — Harness, Loop, Memory, and Eval/LLM-Ops. This context covers the planning agent: a workflow that breaks a goal into multiple steps and runs each one as a real agent turn.

## Language

**Goal**:
The one-sentence task a user hands the planning agent, which never changes once planning starts.

**Plan**:
The current, ordered list of Steps the planning agent intends to run to reach the Goal. A Plan is always the latest truth — it is replaced outright on every Revision, never patched in place.

**Step**:
One unit of the Plan, executed as a single real turn of Waku's agent loop (`run_loop`) against a scoped tool registry. A Step either finishes or fails; it does not pause mid-execution.

**Revision**:
A replacement of the remaining (not-yet-run) Steps in the Plan, triggered when a Step fails or discovers new work. A Revision never touches Steps already finished.
_Avoid_: Replan (the trigger/action) — use Revision for the resulting artifact, replan for the act of producing one.

**Approval gate**:
The point at which the planning agent stops and waits for the user to say go before any Step runs. In Phase 1, a gate fires exactly twice per run at most: once after the initial Plan is produced, and again only if a Revision replaces the remaining Plan — not before every individual Step.
_Avoid_: Per-step approval (the per-step cadence was considered and rejected for Phase 1 — see decision below).

## Decisions

**Approval cadence (Phase 1): approve-once, re-approve-on-revision.**
The planning agent asks for approval after the initial Plan, then runs every Step autonomously without pausing between them — unless a Step triggers a Revision (failure or discovered work), in which case the gate fires again on the new remaining Plan before execution continues. Rejected: approving before every individual Step, which was judged too interruptive for a plan whose steps haven't changed since the last approval.
