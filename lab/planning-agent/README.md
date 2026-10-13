# Planning agent

A LangGraph planner that breaks a goal into steps, runs each one as a Waku loop
turn, revises the plan when it discovers work, and resumes after a crash.

## The question

Does LangGraph's checkpointer earn a place in Waku, or should Waku's own graph
engine grow a smaller one?

## What we connect

Waku's loop (`waku/loop/agent.py:run_loop`) as the executor for each step,
Waku's tool registry for the tools a step may call, and `state.db` for the
dedup table that keeps resume from repeating a side effect. LangGraph supplies
the state machine, the reducers and the checkpointer.

Verified against: langgraph 1.2.15, langgraph-checkpoint-sqlite 3.1.2, 2026-10-12

## Run it

From the repo root:

    pip install langgraph langgraph-checkpoint-sqlite
    python -m pytest lab/planning-agent/tests -q
    python lab/planning-agent/run.py --help

## What we found

Nothing yet. This topic is at the design stage; the design is in
`docs/superpowers/specs/2026-10-11-planning-agent-design.md` and the plan in
`docs/superpowers/plans/2026-10-11-planning-agent.md`.

## Video angle

The hook: a checkpoint tells you what the program knew, not what happened in
the world. The surprising finding: resuming a crashed graph re-runs the failed
node, so a step that booked a meeting and then died books it twice.

## Graduation

Undecided. The question is whether `waku/graph/` should gain a SQLite
checkpointer of its own rather than take the dependency. Phase 4 produces the
evidence.
