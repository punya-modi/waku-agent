# Planning agent — system design

Status: design. No code yet. Scope: a `lab/` topic that explores LangGraph as
the orchestration substrate for long-running, replannable tasks, and reports
what should graduate into `waku/`.

Date: 2026-10-11

---

## 1. What we are adding and why

`waku/loop/agent.py` is one agent turn: the model picks tools until it stops.
`waku/graph/` arranges steps around that loop — fan-out, code routers, bounded
cycles, a whole `run_loop` turn as one node (`nodes.agent_node`).

Three things neither of them expresses:

- **A plan decided at runtime.** A `Graph` is built with fixed nodes before
  `run_graph` runs. A planner decides the steps while the task is underway.
- **A plan that is revised.** A long task discovers work it could not see at
  the start. The step list has to change without losing what is already done.
- **Resume across processes.** A task that runs for hours must survive a
  crash, a restart, and a laptop lid closing.

That third one is named explicitly in
[agent-graphs-design.md](../../agent-graphs-design.md) §2 as the only reason to
revisit the no-LangGraph decision: "Revisit only if graphs become the center of
gravity (checkpointing/resume across processes, distributed nodes). Not now."
This topic is that revisit, run as an experiment rather than as a core change.

## 2. Decision: LangGraph, in `lab/` only

`docs/agent-graphs-design.md` §2 decided against LangGraph for the core engine,
and that decision stands. This topic does not reopen it. It uses LangGraph
where the repo already allows it:

1. **No new dependency, and no change to `pyproject.toml`.** The topic states
   its own `pip install` in its header, which conventions §6 rule 4 allows lab
   code to do: "Use the stdlib or an extra that already exists, or state the
   `pip install` in the file's own header." `lab/memory-native/langmem_native.py`
   is the precedent. Nothing is added to the default install, which keeps hard
   rule 3, and the repo's dependency surface is untouched.

   This matters because the existing `arena` extra is **not sufficient** for
   this topic. `langgraph>=0.3` resolves to 1.2.14, whose runtime dependencies
   are `langchain-core`, `langgraph-checkpoint`, `langgraph-prebuilt`,
   `langgraph-sdk`, `pydantic` and `xxhash`. The SQLite checkpointer is a
   **separate distribution**, `langgraph-checkpoint-sqlite` (3.1.1, which itself
   pulls `aiosqlite` and `sqlite-vec`). Installing `.[arena]` therefore gives
   in-memory checkpointing only, and phase 4 — the kill-and-resume check this
   whole topic rests on — would fail on a fresh environment. `arena` is a
   memory-backend extra in any case, which planning is not.
2. **`lab/` is the designated home.** Conventions §6: "`lab/` holds one folder
   per outside topic: another agent, model or memory product, and how Waku's
   agent, memory and skills connect to it."
3. **Hard isolation.** `lab/` never ships (the wheel packages only `waku`, the
   sdist excludes `/lab/**`), and `make gate` never depends on it, so a
   LangGraph release cannot redden CI.
4. **A graduation path exists.** The topic's README says what, if anything,
   should move into `waku/` and at which rung of the footprint ladder.

The learning goal is explicit: LangGraph's checkpointer is the capability
`waku/graph/` lacks, and the cheapest way to judge whether Waku's engine should
grow one is to use the mature implementation first.

## 3. Boundary

```
lab/planning-agent/
  README.md          # the six playbook headings + Verified against: line
  graph.py           # the LangGraph graph: nodes, edges, reducers
  state.py           # PlanState and the step/revision records
  digest.py          # step digests and phase rollups
  offload.py         # the scratch directory: write, read, search
  idempotency.py     # tool classification and the dedup table
  bridge.py          # execute_step -> waku.loop.agent.run_loop
  run.py             # one command, per the playbook's "Run it"
```

The scratch directory and the dedup table are the two pieces that touch disk.
The scratch directory is per task and may be deleted freely. The dedup table
lives in `state.db`, which is runtime data: hard rule 1 applies, and nothing in
this topic clears it.

Rules this topic lives under, all already enforced:

- **Nothing under `waku/` or `evals/` imports from `lab/`.** The dependency
  runs one way. `lab/planning-agent/` imports from `waku/`; never the reverse.
  `test_product_and_evals_never_reach_into_examples_or_lab` enforces this.
- **`make gate` never depends on this folder.** Its own checks run from
  `run.py`, not from `evals/`.
- **A dated version header.** Conventions §6 rule 5. LangGraph is not installed
  in a default environment, so the first implementation task installs it,
  records the resolved versions in the README's `Verified against:` line, and
  confirms the real import paths for the checkpointer and store, which have
  moved between LangGraph majors. As of 2026-10-11 PyPI serves langgraph
  1.2.14 and langgraph-checkpoint-sqlite 3.1.1; the install line is
  `pip install langgraph langgraph-checkpoint-sqlite`. Note that the `>=0.3`
  floor in `arena` spans a major version boundary, so an existing environment
  and a fresh one can resolve to materially different APIs — which is the
  reason the verified line exists.

The directory is created when implementation starts, not now:
`test_lab_topics_follow_the_playbook` requires a complete playbook and a
verified version the moment the folder exists, and neither is honest yet.

## 4. State: two reducers, one goal

The plan is state, not narrative. A long message history buries the original
ask under dozens of tool calls; a structured plan survives that intact.

```python
class PlanState(TypedDict):
    goal: str                                   # immutable: the original ask
    plan: list[Step]                            # REPLACED by the replanner
    done: Annotated[list[StepResult], add]      # APPEND-ONLY: the update book
    revisions: Annotated[list[Revision], add]   # APPEND-ONLY: why plan changed
    lessons: Annotated[list[Lesson], add]       # APPEND-ONLY: what failed, and why
    rollups: Annotated[list[Rollup], add]       # APPEND-ONLY: finished phases, collapsed
    budget_spent: float                         # USD
    budget_limit: float
```

Two reducers, on purpose:

- **`plan` is overwritten.** It is always the current truth, short enough to
  recite into every step's prompt. One line per step, not a document.
- **Everything else is appended.** `done`, `revisions`, `lessons` and `rollups`
  are the task's event history: what finished, why the plan changed, what was
  tried and failed, and what whole phases amounted to. A thirty-step task stays
  explicable after the fact instead of being a mystery.
- **`goal` never changes.** It is the guard against goal drift, re-injected on
  every step so a long task cannot wander off the original ask.

The records:

| record | carries | why |
|---|---|---|
| `Step` | id, one-line description, status (`pending`, `in_progress`, `done`, `failed`) | the recited plan |
| `StepResult` | step id, prose `digest`, `artifacts` map, `output_path` | what the next step reads |
| `Revision` | what changed, and why a step was added or dropped | the audit trail |
| `Lesson` | the attempt, the failure mode, the diagnosis | stops a replanner re-proposing what just failed |
| `Rollup` | a span of step ids, and one summary of them | keeps the prompt flat past ~20 steps |

`artifacts` is a structured map of identifiers — event ids, file paths, URLs —
kept **verbatim**, separate from the prose `digest`. A digest may legitimately
drop or paraphrase `abc123`; a later step that needs that id needs it exactly.
Digests summarise, artifacts are copied.

## 5. Topology

```
START -> plan -> route (code reads state["strategy"])
                   |
                   +-- deterministic --> execute_step <--+
                   +-- hierarchical ---> decompose ------+
                   +-- orchestrated ---> execute_step ---+
                                              |
                                        replan_gate (code)
                                              |
                   +-- replan needed -> replan -> execute_step
                   +-- steps left ------------> execute_step
                   +-- done or out of budget -> synthesize -> END
```

**The strategy split did not survive implementation.** The topology above is
the design as drawn; the plan collapsed it into one plan-execute-replan cycle,
because the three strategies differ by prompt rather than by topology. Section
13 question 3 anticipated exactly that, and the plan's "Deliberately not in
this plan" records it. There is no `strategy` field and no `decompose` node.

**Control flow is code, never a model.** The planner writes `strategy` and
`plan` into state; plain Python conditional-edge functions read them and pick
the edge. This keeps `agent-graphs-design.md` decision §7.4 ("Routers are code,
not LLM calls — models produce state, functions read it"), so the two engines
stay philosophically compatible and graduation stays possible.

**Replanning is triggered, not scheduled.** `replan_gate` calls the replanner
only when a step failed, or when a step flagged that it discovered new work.
Replanning after every step doubles the LLM spend and buys nothing when the
plan is still correct. When the replanner does run, completed results are
preserved and only the remaining work is restructured.

## 6. Memory: four mechanisms

Waku already compresses within one turn: `waku/loop/trim.py:shrink_seen` cuts
long tool results the model has read, and `waku/memory/reports.py:digest` cuts
a report to its title, summary bullets and key numbers. Both act on the
`messages` list of **one** `run_loop` call. The measured reason they exist, from
`trim.py`: one research turn "sent 626.5k tokens in for 8.4k out: the model cost
$1.334."

Each `execute_step` is a separate `run_loop` call with its own fresh `messages`,
so `shrink_seen` protects the inside of a step and nothing protects the span
across steps. That gap is the second and third mechanisms below.

| Mechanism | Implementation | Scope | Holds |
|---|---|---|---|
| Step working memory | `run_loop` messages + `shrink_seen` | one step | that step's transcript, discarded after |
| Task working memory | `PlanState` | one task | the spine: plan, digests, revisions, lessons, rollups |
| Offload | a per-task scratch directory | one task | full-fidelity outputs, by path, searchable |
| Durable | LangGraph checkpointer; LangGraph store | thread; cross-thread | resume point; facts that outlive the task |

**What the next step sees.** The prompt for step N is: `goal`, the current
`plan` with statuses, the `rollups` for finished phases, the `digests` and
`artifacts` of recent steps, and any `lessons`. Never raw transcripts.

### 6.1 Offload, rather than a blob inside the checkpoint

Full step outputs go to files in a per-task scratch directory; `StepResult`
holds the `output_path`. Two tools are registered in each step's scoped
registry: one to read an earlier output whole, one to search across them. The
model pulls what it needs; the graph does not decide for it.

This is deliberately not "keep the full output in state". A checkpointer
serialises state on every step, so carrying transcripts inside it means write
amplification that grows with the task — the mechanism meant to make resume
fast would get slower at every step. It is also why search matters alongside
read: an index-addressed `read_step_output(n)` only helps a model that already
knows which step holds the detail.

The industry name for this is state offloading: full-fidelity artifacts kept
outside the active context window, referenced from it. It is also the safer
half of the compaction argument — summarising and discarding compounds loss,
while offloading keeps the original and only shortens the prompt.

### 6.2 Lessons: roll back the world, keep the reflection

A failed step's work is discarded. What was learned from it is not. Each
failure appends a `Lesson` — the attempt, how it failed, the diagnosis — and
the replanner is given the lessons along with the plan.

Without this, the replanner is invoked at precisely the moment the state
carries no record of what just went wrong, so it can restructure the remaining
work and re-propose the approach that failed, indefinitely. An agent that keeps
too little of its failures repeats them; one that over-generalises from a
single failure rules out workable alternatives, which is why a `Lesson` records
a diagnosis and not a prohibition.

### 6.3 Rollup: digests bound the step, not the task

A one-line digest per step still grows linearly. Past roughly twenty steps the
digest list alone crowds the prompt, and the degradation is measurable: model
accuracy falls with context length even when every relevant fact is still
present. Per-step digests bound each step's cost; they do not bound the task's.

So a finished phase collapses into one `Rollup` and its individual digests drop
out of the prompt. The underlying outputs stay on disk from §6.1, so a rollup
is a prompt-shortening move, never a loss of information.

### 6.4 Checkpointer and store are different things

Keeping them distinct is half the lesson: the checkpointer is this task's
progress, scoped to a `thread_id`, and makes a killed task resumable; the store
holds facts that outlive the task, across threads.

### 6.5 What a finished task writes back to Waku

One episode for the whole task, and facts only through the existing
`waku/memory/consolidation.py`. A thirty-step task must not write thirty
episodes into `state.db`, and this topic does not reimplement consolidation.

## 7. Idempotency: what resume may re-run

Resume is the point of this topic, and resume is where side effects get
duplicated. A checkpoint records what the *system* knew when it was written; it
cannot record what already happened in the *world*. A step that booked a
meeting and then died before its checkpoint committed looks, on restart, exactly
like a step that never ran.

Waku's tools do not solve this uniformly:

- `waku/tools/calendar.py` already guards `create_event` by natural key —
  "same title+start = same event. A confused model (or an impatient user) must
  not be able to triple-book a meeting."
- `send_message` and `save_note` carry no equivalent guard.
- And the existing guard has a hole a replay walks into. The dedupe check and
  `conn.commit()` run **before** `_write_ics` and the Apple and Google syncs.
  Die in between, and replay hits the guard, returns "already exists (not
  duplicated)", and skips those syncs permanently. The guard makes replay safe
  for the local table by making the downstream writes get skipped instead of
  repeated.

So the planner cannot assume its tools are replay-safe. The contract:

1. **Classify every tool** as read-only or side-effecting. A read-only call is
   free to re-run.
2. **Key every side-effecting call** by `(thread_id, step_id, tool, args_hash)`.
3. **Record the outcome under that key before the step is marked done**, in a
   dedup table in `state.db`. On replay, a hit returns the recorded result
   rather than calling the tool again.
4. **Aim for at-least-once with deduplication**, not exactly-once. Exactly-once
   across a process boundary and a third-party API is not available; a dedup
   window is.

This mirrors what durable-execution engines do by committing a step's output in
the same transaction as the rows it wrote. `state.db` is already a SQLite
database in the right place, so the dedup table is idiomatic rather than new
machinery.

This contract lands as **phase 3, before the resume demonstration in phase 4**.
It is not a later refinement: kill-and-resume is unsafe to run against
side-effecting tools without it.

## 8. Budget

`budget_spent` is in USD, the same unit as the `usage.jsonl` spend ledger, and
accumulates per step; a code gate checks it before each step.
On exhaustion the graph routes to `synthesize`, which answers from what the
task already gathered — the same shape as the loop's guardrail 2, which spends
one tools-off call rather than throwing the turn away.

The digest tier is the real cost control. `budget_limit` only decides when to
stop; digests decide how expensive each step is.

## 9. The bridge to the loop

`execute_step` calls `waku.loop.agent.run_loop` for one sub-task, the same move
`waku/graph/nodes.agent_node` makes: a scoped tool registry, its own working
memory, and the step's observer events passed through for tracing. `run_loop`
is not modified. This is what makes the topic a Waku experiment rather than a
generic LangGraph tutorial, and it is what the graduation assessment will judge.

## 10. How this gets checked

`lab/` is outside `make gate` by rule, so checks run from `run.py` and are
reported in the README's "What we found":

- **Resume is real.** Start a multi-step task, kill the process at a step
  boundary, restart, and assert it continues from the last completed step rather
  than restarting the task. This is the claim the whole topic rests on.
- **Resume does not double-write.** The sharper version of the same test, and
  the one that matters: kill the process *mid-step*, after a side-effecting tool
  has committed but before the checkpoint has. On restart, assert the external
  state was written exactly once. Run it against a tool with no natural-key
  guard, so the dedup table is what is being tested and not
  `create_event`'s own check.
- **A skipped downstream write is caught.** Replay a `create_event` whose local
  row exists but whose calendar-file write did not happen, and assert the
  replay completes it rather than reporting "already exists" and moving on.
- **The update book holds.** Force a step to fail, assert the replanner
  preserves `done`, appends a `Revision` and a `Lesson`, and restructures only
  the remaining work.
- **Lessons are used, not just stored.** Force a step to fail in a way that
  invites the same approach again, and assert the replanner's next plan differs.
  A `Lesson` nothing reads is dead weight.
- **Rollups slow prompt growth; they do not flatten it.** Run a task of at
  least twenty steps and record prompt size per step, with rollups on and off.
  Per-step digests grow linearly. Rollups do not stop the growth, because the
  rollups themselves accumulate: one per eight steps. Measured over 24 steps at
  the bridge's own digest size, the final prompt is 2.2 times smaller with
  rollups on, and growth over the run falls from 18x to 8x. Report that pair.
  Flat growth would need a second compaction tier (a rollup of rollups), which
  this topic does not build and has no measurement to justify.
- **Offload keeps checkpoints small.** Record checkpoint size per step. It
  should not grow with the number of completed steps.
- **Routers stay code.** No LLM call decides an edge.

A behaviour that graduates into `waku/` brings a deterministic eval with it at
that point, per hard rule 4. Nothing in `lab/` is exempt from that on the way in.

## 11. Phasing

Each phase is one component plan, written when it is reached.

| phase | deliverable | proves |
|---|---|---|
| 1 | `state.py` + `graph.py` with stub nodes, no LLM | the topology and the two reducers, at zero spend |
| 2 | `bridge.py`: `execute_step` calls `run_loop` | a real Waku turn as a LangGraph node |
| 3 | `idempotency.py`: tool classification + dedup table | resume cannot double-write |
| 4 | checkpointer + kill/resume checks, including mid-step | the capability `waku/graph/` lacks |
| 5 | replanner + `revisions` + `lessons` | the plan survives discovered work, and does not repeat it |
| 6 | `offload.py` + digests + `artifacts` | details survive without the prompt carrying them |
| 7 | rollups + token and checkpoint-size measurements | long tasks stay affordable |
| 8 | README "What we found" + "Graduation" | the verdict, with numbers |

Phase 3 comes before phase 4 on purpose. The kill-and-resume demonstration is
the headline, but running it against side-effecting tools without the dedup
table in place is how a demo double-books a calendar.

## 12. Decisions taken

1. **LangGraph in `lab/` only.** The core engine's no-LangGraph decision stands.
2. **`plan` replaced, everything else appended.** Two reducers, one state.
3. **`goal` is immutable** and recited every step.
4. **Routers are code**, matching `waku/graph/` decision §7.4.
5. **Replanning is triggered** by failure or discovered work, not scheduled per step.
6. **`run_loop` is untouched**; `execute_step` composes it.
7. **Four memory mechanisms**: step, task, offload, durable.
8. **Full outputs are offloaded to files**, not carried in the checkpoint, and
   are both readable and searchable.
9. **`artifacts` are verbatim, digests are prose.** Identifiers are never
   summarised.
10. **A failure appends a `Lesson`**, and the replanner reads it. Roll back the
    work, keep the reflection.
11. **Finished phases collapse into a `Rollup`**, which cuts the prompt by
    roughly 2.2x over 24 steps rather than flattening it: rollups accumulate
    too, one per eight steps.
12. **Side-effecting tools get an idempotency key and a dedup table** in
    `state.db`; at-least-once with deduplication, not exactly-once. This lands
    before the resume demonstration, not after.
13. **One episode per finished task**, facts via existing consolidation.
14. **The lab folder is created at implementation time**, with a verified version.
15. **The topic declares its own `pip install`** rather than amending the `arena`
    extra, which does not carry the SQLite checkpointer and is scoped to memory
    backends.

## 13. Open questions

1. **Does the checkpointer earn a place in `waku/graph/`?** The graduation
   question. Phase 4 produces the evidence; the answer may be "copy the idea,
   not the dependency" — a SQLite checkpointer for Waku's own engine is a small
   file, and `state.db` already exists.
2. **Where planning state would live after graduation.** LangGraph's
   checkpointer wants its own SQLite file; Waku keeps one `state.db`. If this
   graduates, the decision is whether plan progress becomes a table there.
3. **Whether the strategy split survives contact.** `deterministic`,
   `hierarchical` and `orchestrated` may collapse into one plan-execute-replan
   cycle with different prompts. Phase 1 should be built so that collapsing them
   is cheap.
