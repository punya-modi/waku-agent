# Planning agent — spec

Tracker spec for the `lab/planning-agent` topic. The system design is in
`2026-10-11-planning-agent-design.md`; the task breakdown is in
`docs/superpowers/plans/2026-10-11-planning-agent.md`. This file is the parent
issue body that the tickets hang off.

## Problem Statement

Waku can run one agent turn well. It cannot run a task that takes hours.

Three things are missing, and each one bites a person trying to use Waku for
real work:

- **The steps have to be known in advance.** A graph is built with fixed nodes
  before it runs, so a task whose shape depends on what it finds cannot be
  expressed. Someone asking Waku to organise an event, migrate a dataset or
  research a decision has to decompose the work themselves, in advance, and
  guess right.
- **A plan cannot change its mind.** A long task discovers work it could not
  see at the start. Today that means abandoning the turn and starting again,
  losing everything already done.
- **A crash loses the task.** A process that dies, a restart, or a closed
  laptop lid takes hours of progress with it. Worse, restarting a task that
  already sent a message or booked a meeting does those things again, because
  nothing records what already happened in the world as opposed to what the
  program knew.

The third one is the reason this work exists now. It is named in
`agent-graphs-design.md` section 2 as the single condition that would justify
revisiting the decision not to use LangGraph: "Revisit only if graphs become
the center of gravity (checkpointing/resume across processes, distributed
nodes). Not now." This is that revisit, run as an experiment rather than as a
change to the core engine.

## Solution

A planning agent, built as a `lab/` topic on LangGraph, that decomposes a goal
into steps, runs each step as one ordinary Waku loop turn, revises the plan
when it discovers work, and resumes after a crash without repeating anything it
already did to the world.

From the point of view of someone running it: give it a goal and a budget in
dollars. It writes a plan, works through it, and tells you what it did. If a
step fails it records what went wrong and restructures the rest of the work
rather than giving up. If the process dies, starting it again picks up where it
left off, and a meeting it already booked stays booked once. If it runs out of
budget it answers from what it has instead of stopping dead.

The topic also answers a maintainer's question, which is why it lives in `lab/`
and not in `waku/`: does Waku's own graph engine need LangGraph, or does it
need a checkpointer? The topic produces measured evidence and a verdict.

## User Stories

1. As someone running a long task, I want to give Waku a goal and have it work
   out the steps itself, so that I do not have to decompose the work in advance
   and guess right.
2. As someone running a long task, I want the original ask re-stated to the
   agent on every step, so that a thirty-step task does not drift into
   answering a different question than the one I asked.
3. As someone running a long task, I want to see the current plan with each
   step's status, so that I can tell what has happened and what is coming.
4. As someone running a long task, I want a step that discovers new work to
   cause the plan to be revised, so that the task adapts instead of completing
   a plan that is now wrong.
5. As someone running a long task, I want work that is already finished to be
   preserved when the plan is revised, so that a revision costs me the
   remaining work and not the whole task.
6. As someone running a long task, I want a written record of why the plan
   changed, so that I can understand after the fact why the task did what it
   did.
7. As someone running a long task, I want a failed step to record what was
   tried and why it failed, so that the revision does not propose the approach
   that just failed.
8. As someone running a long task, I want a step that keeps failing to be
   abandoned rather than retried indefinitely, so that one impossible step does
   not consume the whole budget.
9. As someone running a long task, I want the task to terminate even when
   failures are free, so that a bad API key does not spin the task until a
   recursion limit stops it.
10. As someone running a long task, I want to set a budget in dollars, so that
    I can bound what an unattended task costs me.
11. As someone running a long task, I want the budget to be checked before a
    step runs rather than after, so that a budget of zero buys nothing.
12. As someone running a long task, I want budget exhaustion to produce an
    answer from what was gathered, so that spending the budget is not the same
    as wasting it.
13. As someone running a long task, I want the process to be killable at any
    moment, so that I can close my laptop without losing hours of progress.
14. As someone running a long task, I want a restarted task to continue from
    the last completed step, so that resuming does not redo work I already paid
    for.
15. As someone running a long task, I want a side effect that already happened
    not to happen again on resume, so that a restart cannot double-book a
    meeting or send a message twice.
16. As someone running a long task, I want a failed tool call to stay
    retryable, so that a timeout recorded during a crash does not become a
    permanent failure for the rest of the task.
17. As someone running a long task, I want two separate tasks doing the same
    thing to both actually do it, so that deduplication never silently skips a
    real action.
18. As someone running a long task, I want each step to read short summaries of
    earlier steps rather than their transcripts, so that step thirty does not
    cost thirty times step one.
19. As someone running a long task, I want identifiers produced by a step kept
    exactly as written, so that a later step that needs an event id gets the
    id and not a paraphrase of it.
20. As someone running a long task, I want the full output of every step kept
    on disk, so that summarising never loses something I needed.
21. As someone running a long task, I want the agent able to read an earlier
    step's full output when it needs it, so that a summary is a shortcut rather
    than a wall.
22. As someone running a long task, I want the agent able to search across
    earlier outputs, so that it can find a detail without already knowing which
    step produced it.
23. As someone running a long task, I want finished phases collapsed into one
    summary, so that the prompt does not grow without bound as the task gets
    longer.
24. As someone running a long task, I want the full outputs kept out of the
    saved progress, so that saving progress does not get slower at every step.
25. As a maintainer, I want every routing decision made by code rather than by
    a model, so that the two graph engines stay philosophically compatible and
    graduation remains possible.
26. As a maintainer, I want each step to be one unmodified Waku loop turn, so
    that this measures Waku composed with a planner rather than a generic
    LangGraph tutorial.
27. As a maintainer, I want the experiment to add no default dependency, so
    that someone installing Waku is unaffected by it.
28. As a maintainer, I want the topic unable to redden CI, so that a LangGraph
    release cannot break the repo's gate.
29. As a maintainer, I want a measured answer to whether resume works across a
    real process boundary, so that the headline claim is evidence and not an
    assertion.
30. As a maintainer, I want the prompt-growth claim measured with numbers, so
    that the finding reports what the design actually achieves rather than what
    it hoped for.
31. As a maintainer, I want the size of saved progress measured per step, so
    that the decision to keep outputs on disk is justified rather than assumed.
32. As a maintainer, I want a stated verdict on whether the checkpointer should
    graduate into Waku, and at which rung of the footprint ladder, so that the
    experiment ends in a decision rather than a maybe.
33. As a maintainer, I want findings about Waku's own tools recorded separately
    from the LangGraph verdict, so that a bug found along the way is actionable
    whatever happens to this topic.
34. As a contributor new to the repo, I want the topic to state the exact
    versions it was verified against, so that I can tell whether a failure is
    mine or the library's.
35. As a contributor new to the repo, I want one documented command that runs
    the topic's checks, so that I can see it work without reading the plan.
36. As a contributor new to the repo, I want that command to work on a clean
    shell, so that the documented entry point is not broken by an import
    problem.
37. As a contributor on a default install, I want the topic's tests to skip
    rather than fail, so that not having LangGraph installed does not look like
    a broken repo.
38. As a coding agent implementing this, I want every node injected as a
    callable, so that I can test the whole topology without a model, a key or a
    network.
39. As a coding agent implementing this, I want each step's identifier
    validated before it reaches a filename, so that a step id cannot write
    outside the scratch directory.
40. As a coding agent implementing this, I want the next step selected by
    identity rather than position, so that a revision that shortens or reorders
    the plan cannot re-run a finished step or index past the end.

## Implementation Decisions

**LangGraph, in the lab only.** The core engine's decision not to adopt
LangGraph stands and is not reopened. The topic states its own install line
rather than amending any extra, because the existing `arena` extra carries
LangGraph but not the SQLite checkpointer, and is scoped to memory backends in
any case. Nothing is added to the default install.

**Modules and their roles.** Nine modules, each one thing: the state records
and next-step selection; the graph topology and its code routers; the bridge
from one planned step to one Waku loop turn; the scratch directory that holds
full outputs; the idempotency contract and its dedup table; deterministic
digests and phase rollups; the replanner; the two database handles and the
thread config; and a command that runs the topic's own checks. A tenth, tiny
module exists only to make the hyphenated topic directory importable, and is
loaded by file path because the hyphen blocks importing anything in that
directory, itself included.

**State: one replaced field, five appended ones.** The plan is always current
truth and is overwritten wholesale. Everything else accumulates and is never
rewritten. The goal is immutable. This shape is the design, so it is given
precisely rather than described:

    goal: str                                   # immutable: the original ask
    plan: list[Step]                            # REPLACED by the replanner
    done: Annotated[list[StepResult], add]      # APPEND-ONLY: the update book
    revisions: Annotated[list[Revision], add]   # APPEND-ONLY: why plan changed
    lessons: Annotated[list[Lesson], add]       # APPEND-ONLY: what failed, why
    rollups: Annotated[list[Rollup], add]       # APPEND-ONLY: phases collapsed
    budget_spent: float                         # USD
    budget_limit: float
    replans: int                                # replaced; the hard bound

The append-only reducer on `done` is what makes it impossible for a revision to
rewrite history even by mistake. It is also why a rollup cannot remove anything
from `done`: the prompt builder has to skip steps a rollup already covers, and
a measurement that edits `done` directly measures a system no graph can run.

**Topology.** Plan, then a bounded execute cycle, then one synthesis. Four
nodes and a rollup node. Every edge is chosen by a plain function reading
state; no model decides an edge. The strategy split in the design document
(deterministic, hierarchical, orchestrated, with a decompose node) collapsed
into this one cycle, because the three differ by prompt rather than by
topology. There is no strategy field.

**Replanning is triggered, not scheduled.** It runs when a step failed or
flagged discovered work. Running it after every step doubles the spend and buys
nothing while the plan is still correct.

**Two termination bounds, not one.** A step with two lessons against it is
skipped. The whole task caps total replans. Both are needed: the per-step bound
is evaded by a replanner that renames the step that keeps failing, and the
budget is not a bound at all, because a step whose loop raises before its first
model call costs nothing.

**The replan edge is conditional.** The router answers "replan" before it
checks the budget, so an unconditional edge back into the cycle would buy one
step past the limit. The edge reuses the same gate that guards the first step.

**Four memory mechanisms, at four scopes.** Within one step, Waku's existing
trimming cuts tool results the model has already read. Across the task, the
state carries the spine: plan, digests, revisions, lessons, rollups. Full
outputs are offloaded to a per-task scratch directory, both readable and
searchable, so a digest is a shortcut and not a loss. Durable progress is the
checkpointer's, keyed to a thread.

**Digests are prose, artifacts are verbatim.** A digest may legitimately
paraphrase; an identifier may not. They are separate fields for that reason.
Digests and rollups are deterministic and call no model, so they cost nothing
and tests can pin exact strings.

**Idempotency: at-least-once with deduplication.** Exactly-once across a
process boundary and a third-party API is not available. Every side-effecting
call is keyed by thread, step, tool and arguments; read-only tools are named
and everything unknown is assumed to change the world, because the other
default means a newly added tool silently skips the guard. The thread is part
of the key so two tasks doing the same thing both do it.

**Every outcome is recorded; only successes are replayed.** Waku's tool
registry returns a tool's exception as an ordinary string, so a failure is not
distinguishable from a result by type. A table that stores results without a
status replays a transient timeout for ever. Rows carry `ok` or `error`; a
replay honours `ok` and re-runs `error`.

**The guard sits on the tool registry, not on the bridge.** The loop executes
tools itself, so a dedup table the bridge merely holds would never be
consulted. The schemas the model sees are unchanged, so this is invisible from
the prompt's side.

**Two databases, two lifetimes.** The checkpointer owns its own file and may be
deleted with the task. The dedup table lives in Waku's existing state database,
because the record of what was already done to the world must outlive any
single task. Nothing in the topic drops or recreates either; a stale schema is
reported, never repaired.

**The recursion limit is set deliberately.** The library's resolved default is
in the thousands, which is not a guard. A planned task needing more than a
couple of hundred supersteps has gone wrong, and stopping is the right answer.

## Testing Decisions

**What makes a good test here.** It asserts something a user or a maintainer
would notice, through the module's public surface, and it fails for one reason.
It does not reach into internals, does not assert on prompt wording beyond the
facts that must be present, and does not need a key, a network or a model. The
strongest tests in this set are the paired ones: the same scenario run with and
without a mechanism, so the test states what the mechanism is for rather than
merely exercising it.

**Three seams, one of them primary.** The graph builder takes its four node
functions and its checkpointer as parameters, so every structural property —
both routers, the budget gate, the replan cap, the conditional replan edge,
cycle termination, resume, checkpoint size — is tested with scripted nodes at
zero spend. The bridge takes the loop function as a parameter, and the
replanner takes its proposer as one; those two are the only points where a
model is reachable, and substituting them is what keeps the whole suite
offline. Nothing else needs a seam: the routers, next-step selection, digests,
rollups, key composition and artifact parsing are pure functions called
directly.

**Prior art.** The injected-callable shape is taken from the existing triage
workflow, whose own docstring says the callables are injected so evals can
script them. The testing idiom is taken from the graph engine's deterministic
evals, which open by declaring pure-function nodes, zero LLM and zero network,
and script nodes as lambdas. Note that no existing lab topic has tests at all,
so this is the first; the prior art is in the deterministic suite, not in the
lab.

**What gets tested, module by module.** The two reducers, including a
deliberate demonstration that removing the appending reducer makes the test
fail. Next-step selection against a revision that shortens, reorders or empties
the plan, and against an exhausted step. Digest and rollup output, pinned to
exact strings. The scratch directory's write, read and search, and refusal of
an unsafe identifier, with a demonstration that loosening the guard lets an
escape through. Key composition, including argument order and thread
separation. The dedup table's replay, its refusal to replay a failure, and its
refusal of a stale schema. Both routers at every branch. The bridge's prompt
contents, its digest and artifact merge, and its behaviour when the loop
raises. The replanner's preservation of finished work, its revision record, its
use of lessons, and its behaviour when the proposer itself fails.

**The claim the topic rests on gets a real process boundary.** Most resume
tests open a second database connection in the same interpreter, which is
close enough to a restart and cheap enough to run every time. One test does it
properly: a child interpreter commits a side effect and then exits without
unwinding, before any progress is saved, and a second child resumes and asserts
the external state was written exactly once. It is the slowest test in the
topic and the only one that earns the wording the README uses.

**The measurements are tests, not prose.** Prompt size with and without
rollups, and saved-progress size per step, are asserted as ratios with
headroom, and the measurement reads its digest size from the module the real
prompt path uses. A measurement that drifts from the real path measures
nothing, which already happened once: digesting at a fifth of the real size
made the mechanism look worthless when it was the measurement that was wrong.

**Isolation.** Every test module skips itself when LangGraph is absent, because
the repo has no pytest configuration and a bare run from the root discovers the
lab. No lab test is placed in the eval suites, so the gate never depends on
this topic.

## Out of Scope

- **The cross-thread store.** The design lists it for facts that outlive a
  task, then routes durable facts through Waku's existing consolidation. Both
  cannot be true, and Waku's memory is the one that already exists. A second
  durable store beside the existing one would be two answers to one question.
- **Writing a finished task back into Waku's memory.** One episode per finished
  task is wanted, but what that episode says and when it is written is not
  settled. It needs a decision before it needs a ticket.
- **The strategy split and the decompose node.** Collapsed, as above. Adding
  either back is a design change.
- **A second compaction tier.** Rollups cut the prompt by roughly a factor of
  two over twenty-four steps; they do not flatten growth, because rollups
  accumulate. Flat growth would need a rollup of rollups, and there is no
  measurement past twenty-four steps to justify compounding the summarising
  loss.
- **Fixing the tool bug this work uncovered.** The existing calendar guard
  commits its local row before writing the calendar file and syncing, so a
  replay is told the event already exists and those writes never happen. That
  is a bug in Waku on its own terms and belongs in its own issue, not here.
- **Changing the loop.** The loop is composed, never modified.
- **Graduating anything.** This topic produces a verdict. Acting on it is
  separate work, and anything that moves into the product brings a
  deterministic eval with it at that point.

## Further Notes

The topic lives under `lab/`, which never ships and is never part of the gate,
and it carries the six playbook headings and a verified-version line from its
first commit, because the repo's checks require them the moment the directory
exists.

The work is ordered so that the idempotency contract lands before the
resume demonstration. The demonstration is the headline, but running it against
side-effecting tools without the dedup table in place is how a demo
double-books a calendar.

Two findings are expected to be worth more than the verdict itself. The first
is that resuming re-runs the whole failed node, so a step that commits a write
and then dies commits it twice; this is measured, not theoretical. The second
is that a checkpoint records what the program knew, never what happened in the
world, which is the reason the dedup table exists at all.
