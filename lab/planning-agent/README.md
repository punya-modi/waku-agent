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

Measured on langgraph 1.2.15 and langgraph-checkpoint-sqlite 3.1.2 with stub
nodes, so these are properties of the structure and not of a provider on the
day.

- **Resume works across a process boundary.** A run killed with `os._exit(1)`
  in step 2 of 3, before any checkpoint for that step, picked up at step 2 in
  a second interpreter and did not repeat step 1.
- **Resume re-runs the failed node, so a side effect commits twice.** The same
  three-step run left `['a','b','b','c']` in an external list while its
  checkpointed state read `['a','b','c']`. A dedup table keyed by thread, step,
  tool and arguments brings the world back to `['a','b','c']`.
- **Waku's own guards are uneven.** `create_event` dedupes by title and start;
  `send_message` and `save_note` do not. And `create_event`'s guard runs before
  the ics write and the Apple and Google syncs, so a replay is told "already
  exists (not duplicated)" and those syncs never happen.
- **Rollups slow prompt growth; they do not flatten it.** 16,241 chars at the
  last of 24 steps without rollups, 7,419 with, a ratio of 2.19x. Growth over
  the run falls from 17.9x to 8.2x, and the prompt with rollups on still
  doubles from 12 steps to 24 (3,574 to 7,419, which is 2.08x). Flat growth
  would need a second compaction tier, a rollup of rollups, which this topic
  did not build: there is no measurement past 24 steps to justify compounding
  the summarising loss.
- **A rollup only shortens a prompt if the digests it replaced leave.** `done`
  is append-only, so nothing can be removed from it; the prompt builder has to
  skip the steps a rollup already covers. Built the other way, a rollup makes
  the prompt longer, and a measurement that edits `done` directly will not
  show it.
- **Checkpoint size** did not grow with the number of finished steps once full
  outputs moved to the scratch directory. Over 12 steps the state gained about
  5,000 chars in total: the digests, the plan and one rollup, where a 4KB step
  output carried in state would have been 48,000.
- **The budget is not a termination bound.** A step whose loop raises before
  its first model call costs nothing, so `budget_spent` never moves and the
  step is selected again indefinitely. Two bounds fixed it: a step with two
  lessons against it is skipped, and the task caps total replans at five. The
  per-step bound alone is evadable, because a replanner that renames a failed
  step resets its lesson count.
- **A dedup table has to record failures as failures.** `ToolRegistry.execute`
  returns a tool's exception as an ordinary string, so a table that stores
  results without a status replays a timeout for ever. Recording every outcome
  with `ok` or `error`, and replaying only `ok`, keeps the crash window closed
  while leaving a transient failure retryable.

The two findings worth more than the verdict are the second and the last: a
checkpoint records what the program knew, never what happened in the world,
and the table that closes that gap is only correct if it can tell a failure
from a result.

## Video angle

The hook: a checkpoint tells you what the program knew, not what happened in
the world. The surprising finding: resuming a crashed graph re-runs the failed
node, so a step that booked a meeting and then died books it twice.

## Graduation

**Copy the idea, not the dependency.** `waku/graph/` should grow its own
SQLite checkpointer rather than take LangGraph. Nothing this topic needed from
LangGraph is large: a `thread_id`, a serialised state blob, a step cursor and
the two reducers. The checkpointer is the part that earns its keep, and it is
about a hundred lines against a dependency tree of roughly forty packages,
`langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core`, `langsmith`,
`pydantic`, `httpx`, `orjson` and `zstandard` among them. Hard rule 3 says the
core is stdlib plus the two model clients, and this is not the capability that
should break it.

The piece lands on **rung 1 of the footprint ladder**: extend something that
already exists, as a checkpointer behind the `waku/graph/` interface. It is not
a rung at all if it arrives as a top-level package, which is an architecture
decision needing a proposal (conventions section 2) — and it does not need to
be one, because `waku/graph/` is already there.

What should NOT be copied is the dedup table's place in the design. That finding
is independent of LangGraph: `create_event` commits its row, then writes the ics
file and runs the Apple and Google syncs, and its natural-key guard sits before
all three. Any replay — a resumed graph, a retried gateway, a user pressing send
twice — is told "already exists (not duplicated)" and the syncs never run. That
is a `waku/` bug on its own terms and is worth an issue on `upstream` whatever
happens to this topic.
