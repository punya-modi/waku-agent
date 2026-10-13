# Planning agent: the architecture, module by module

This is the long form of the topic README. The README states the question, the
findings and the verdict; this file explains how the thing is built and why
each piece is shaped the way it is.

Written 2026-10-13, against langgraph 1.2.15 and langgraph-checkpoint-sqlite
3.1.2. Nine modules plus an import shim: 957 lines of implementation, 1,188 lines of
test, 86 tests.

## Contents

1. [What it does](#1-what-it-does)
2. [The shape of the system](#2-the-shape-of-the-system)
3. [The state, and why the reducers are the design](#3-the-state-and-why-the-reducers-are-the-design)
4. [The graph and its two routers](#4-the-graph-and-its-two-routers)
5. [The bridge: one step is one Waku loop turn](#5-the-bridge-one-step-is-one-waku-loop-turn)
6. [Memory: four mechanisms over three spans](#6-memory-four-mechanisms-over-three-spans)
7. [Idempotency: the part that is not about LangGraph](#7-idempotency-the-part-that-is-not-about-langgraph)
8. [Replanning](#8-replanning)
9. [Sessions, resume and the two databases](#9-sessions-resume-and-the-two-databases)
10. [Termination: four bounds, and why one is not enough](#10-termination-four-bounds-and-why-one-is-not-enough)
11. [The hyphen problem](#11-the-hyphen-problem)
12. [Where it touches Waku](#12-where-it-touches-waku)
13. [The test suite](#13-the-test-suite)
14. [Measurements](#14-measurements)
15. [What is deliberately absent](#15-what-is-deliberately-absent)
16. [Known limits](#16-known-limits)
17. [File map](#17-file-map)

---

## 1. What it does

Given a goal in one sentence, the agent writes a plan of steps, runs each step
as one turn of Waku's own agent loop, revises the plan when a step discovers
work or fails, and answers once. If the process dies halfway through, the next
run picks up where it stopped without redoing finished steps and without
repeating side effects that already reached the world.

Four capabilities, each with a module behind it:

| Capability | Where it lives |
|---|---|
| decompose a goal, revise the decomposition | `state.py`, `replan.py` |
| run one step as a real agent turn | `bridge.py` |
| survive a crash | `session.py`, the SQLite checkpointer |
| survive a crash *safely* | `idempotency.py` |

The fourth is the one worth the topic. The first three are what LangGraph is
for; the fourth is what LangGraph does not do and cannot do, because it is not
a fact about the program.

## 2. The shape of the system

```
                    ┌─────────────────────────────────────────┐
                    │  PlanState (one TypedDict, checkpointed)│
                    │  goal · plan · done · rollups · lessons │
                    │  revisions · budget · replans           │
                    └─────────────────────────────────────────┘
                                      ▲ reducers merge node returns
                                      │
  START ─► plan ─┬─(after_plan)─► execute_step ─► roll_up ─┬─(after_step)─┐
                 │                      │                 │              │
                 │                      ▼                 ├─► replan ────┤
                 │                 run_loop               │   (after_plan)
                 │                  (Waku)                └─► execute_step
                 │                      │
                 │                      ▼
                 │              Scratch (disk)  ──  DedupTable (state.db)
                 │
                 └──────────────────► synthesize ─► END
```

Three rules hold the shape together.

**Models write state; functions read it and pick the edge.** `after_plan` and
`after_step` are ordinary Python predicates over `PlanState`. No LLM call ever
decides an edge. This is the rule `waku/graph/` already follows, and it is why
the whole topology is testable at zero spend: `tests/test_graph.py` exercises
14 routing cases without a network.

**Every node is injected.** `build_graph` takes `plan_fn`, `execute_fn`,
`replan_fn` and `synthesize_fn` as keyword arguments, the same shape
`waku/graph/workflows/triage.py` uses for `build_triage_graph`. Tests pass
stubs; the real caller binds a client at the edge. Nothing in `graph.py`
imports a provider.

**Big things live outside state.** A checkpointer re-serialises the whole state
on every superstep, so a 4KB transcript carried in state is rewritten on every
later step — quadratic writes for a linear task. Full step outputs go to disk
(`Scratch`), and state carries a 600-char digest plus the path.

## 3. The state, and why the reducers are the design

`state.py` defines five record types and one `TypedDict` for the state itself.
The reducer annotation on each field is the whole design decision:

```python
class PlanState(TypedDict):
    goal: str                                              # never changes
    plan: list[Step]                                        # REPLACED on write
    done: Annotated[list[StepResult], operator.add]         # APPENDED
    revisions: Annotated[list[Revision], operator.add]      # APPENDED
    lessons: Annotated[list[Lesson], operator.add]          # APPENDED
    rollups: Annotated[list[Rollup], operator.add]          # APPENDED
    budget_spent: float
    budget_limit: float
    replan_requested: bool
    last_error: str
    replans: int                                            # REPLACED
```

`plan` carries no reducer, so every write replaces it outright. It is always
the current truth, short enough to recite into each step's prompt, and the
replanner can restructure it freely.

The four `operator.add` fields are the task's event history. Nothing in them is
ever rewritten, which is not a convention but a mechanical guarantee: **the
replanner cannot rewrite history even by mistake**, because the reducer
provides no way to remove an element. That guarantee is what makes
`replan.py` trustworthy without a single defensive check.

It also forces one thing that is easy to get wrong. A rollup is supposed to
*shorten* the prompt by collapsing eight finished steps into one summary — but
it cannot remove those eight entries from `done`. So the collapse has to happen
in the prompt builder, by skipping any `done` entry a rollup already covers
(`bridge._history`). Build it the other way and the rollup is pure addition:
the prompt gets **longer**, and every size measurement still passes. One test
pins this by asserting a rolled-up digest is *gone* from the prompt while an
unrolled one stays.

`goal` never changes and is re-injected into every step prompt. That is what
stops a twenty-step task drifting off the original ask.

### Selecting the next step

```python
def next_step(state) -> Step | None:
    finished = finished_ids(state)
    for step in state["plan"]:
        if step["id"] in finished:              continue
        if attempts(state, step["id"]) >= MAX_STEP_ATTEMPTS: continue
        return step
    return None
```

Selection is **by id, never by position**. The obvious implementation,
`state["plan"][len(state["done"])]`, breaks the moment a replan reorders or
shortens the plan: `len(done)` can legitimately exceed `len(plan)`, and the
index then points past the end or at a step that already ran. Three tests cover
the shrink case alone.

Two helpers, `finished_ids` and `rolled_up_ids`, exist because four call sites
across three modules asked the same two questions and each had its own set
comprehension. One definition each, in `state.py`.

## 4. The graph and its two routers

```python
def after_plan(state) -> str:        # also the edge out of replan
    if state["budget_limit"] <= 0 or _out_of_budget(state): return "synthesize"
    if next_step(state) is None:                            return "synthesize"
    return "step"

def after_step(state) -> str:        # the edge out of roll_up
    if state["replan_requested"] and state["replans"] < MAX_REPLANS: return "replan"
    if _out_of_budget(state):                               return "synthesize"
    if next_step(state) is None:                            return "synthesize"
    return "step"
```

Three subtleties, each with its own test.

**An empty plan answers rather than spins.** If the planner returns no steps,
`after_plan` routes straight to `synthesize`. Without that, the cycle either
spins or crashes on a `None` step.

**A budget of zero or less halts before any step runs.** The gate sits on the
edge *out of* `plan`, not after the first step, so a zero limit never buys one
step's worth of work.

**The replan edge cannot buy a step.** This is the non-obvious one.
`after_step` answers `"replan"` *before* it checks the budget, deliberately: a
step that failed leaves an untrustworthy plan whether or not money is left. But
that means a plain `replan ─► execute_step` edge would spend one step past the
limit. So the edge out of `replan` is conditional, through `after_plan`, which
already asks exactly the right three questions. A test drives a graph whose
step exhausts the budget and asks for a replan, and asserts `done == []` with
`replans == 1`: the replan ran, and then the budget stopped it.

The `roll_up` node sits between `execute_step` and the router, so every step
passes through it and `after_step` reads the same state one node later.

## 5. The bridge: one step is one Waku loop turn

`bridge.py` is the join, and the reason this is a Waku experiment rather than a
LangGraph tutorial. `make_execute_step` returns the `execute_step` node, closed
over a client, a model, a registry, a `Scratch` and optionally a `DedupTable`.

`run_loop` is a **parameter, not an import**. Tests substitute a fake and never
touch a network; the real caller passes `waku.loop.agent.run_loop`. `run_loop`
itself is not modified, wrapped or subclassed.

One step does five things:

1. Pick the step (`next_step`); return `{}` if nothing is left.
2. Scope the registry — if a dedup table is present, wrap the side-effecting
   tools (§7).
3. Build the prompt and call `run_loop` with a `UsageMeter` as observer and
   `shrink_seen` as the trim function.
4. Write the full reply to `Scratch`.
5. Merge back a `StepResult`: a digest, the verbatim artifacts, and the path.

### The prompt

```
You are working through one step of a larger task.

The overall goal, which never changes:
{goal}

The plan as it stands:
1. [done] find a venue
2. [pending] book it

What has happened so far:
- earlier phase (s0, s1, ...): {rollup summary}
- a: three venues found [shortlist=v1,v2]
- lesson from b: tried calling the venue, line was dead. use email

Your step now: book it

When your step produces identifiers that a later step will need exactly - an
event id, a file path, a URL - end your reply with one line:
ARTIFACTS: key=value; key=value
```

It carries the goal, the plan, the rollups, the unrolled digests with their
artifacts, and the lessons. It never carries a transcript.

The `[done]` / `[pending]` marker is **derived from `done`**, not read off the
`Step` record. Nothing writes a `Step`'s own `status` field after the planner
sets it, so reciting it printed `[pending]` against work the model had already
finished — found in review, fixed, and worth stating because the record type
still carries the field.

### Artifacts

A digest may paraphrase; an event id must survive exactly. So a step can end
its reply with a marker line of `key=value` pairs, kept verbatim in
`StepResult.artifacts` and reprinted in later prompts. The parser is
deliberately forgiving: a pair with no `=` is skipped, never fatal, because the
line is written by a model and is therefore input.

### Cost

`UsageMeter` is the observer `run_loop` already calls as `notify(kind, event)`.
It reads the `llm` events' `{"usage": {"in": N, "out": N}}` and converts to
dollars at prices quoted per million tokens — the same unit providers quote and
the same unit as Waku's `usage.jsonl` ledger. Non-`llm` events are ignored.
The step adds `meter.cost` to `budget_spent`, which is what the routers gate on.

### Failure

A `run_loop` that raises does not crash the graph. The step returns
`replan_requested=True`, a `Lesson` naming what was attempted and what failed,
`done: []` and the accumulated cost. The graph then decides what happens next —
which is the point of keeping control flow in code.

## 6. Memory: four mechanisms over three spans

Context management here is not one trick but four, each covering a span the
others cannot see:

| Span | Mechanism | Where |
|---|---|---|
| inside one step | `shrink_seen` cuts tool results the model has read | `waku/loop/trim.py` (existing) |
| step to step | a 600-char digest stands in for a whole output | `digest.step_digest` |
| phase to phase | eight digests collapse into one 800-char rollup | `digest.make_rollup` + `graph.maybe_roll_up` |
| any step, on demand | the full output is on disk, readable and searchable | `offload.Scratch` |

Only the first already existed in Waku. The other three are what this topic
adds, and they are what the measurements in §14 are about.

`Scratch` is one file per step under a per-task directory. The step id reaches
a filename, so it is validated against a **closed character set** —
`[A-Za-z0-9_-]{1,64}`, letters, digits, hyphen and underscore allowed and
everything else refused, on both write and read. An id containing `/` or `..`
would otherwise steer the write outside the directory. Loosening the pattern to
`.+` fails seven tests, `../escape` among them.

## 7. Idempotency: the part that is not about LangGraph

This is the finding worth more than the verdict.

**A checkpoint records what the program knew. It cannot record what already
happened in the world.** Resuming a crashed graph re-runs the whole failed
node, so a step that committed a write and *then* raised commits it twice. This
is measured, not theoretical: a three-step run crashing in step `b` leaves

```
world:      ['a', 'b', 'b', 'c']      ← the outside world
checkpoint: ['a', 'b', 'c']           ← what the program thinks happened
```

The state is perfectly consistent. The world is not.

`idempotency.py` closes it with a table in `state.db`:

```sql
CREATE TABLE planning_tool_calls (
  key TEXT PRIMARY KEY,     -- sha256(thread_id, step_id, tool, sorted args)
  result TEXT NOT NULL,
  status TEXT NOT NULL,     -- 'ok' | 'error'
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)
```

Five decisions in that schema and the code around it:

**`thread_id` is in the key.** Without it, two different tasks that
legitimately perform the same action collide and the second one is silently
skipped — a real booking that never happens. One test runs the same call under
`task-1` and `task-2` and asserts both act.

**Read-only tools are named; everything else is assumed to change the world.**
`READ_ONLY` is a closed set. A tool written next year is deduped until someone
decides it needs no guard. Guessing the other way round means a new tool
silently skips the table. Emptying `READ_ONLY` fails two tests, because a step
that lists events twice would see the first answer replayed.

**A failure is recorded as a failure and not replayed.** `ToolRegistry.execute`
catches a tool's exception and returns `f"Error running {name}: {exc}"` as an
ordinary string, so a caller cannot tell a failure from a result by type. A
table that stores results without a status replays a transient timeout for
ever. Here every outcome is recorded, only `status='ok'` is a replay hit, and
the retry reaches the tool. `setup()` refuses a table that predates the
`status` column rather than silently using it — and reports it rather than
migrating it, because it is runtime data under hard rule 1.

**The guard sits on the registry, not on the bridge.** `run_loop` executes
tools itself, through `ToolRegistry.execute`. A dedup table the bridge merely
held would never be consulted. So `guarded_registry` builds a new registry
whose side-effecting tools are wrapped:

```python
guarded.register(replace(tool, fn=wrapped))   # Tool is a dataclass
```

`dataclasses.replace` keeps `wants_notify` and everything else, and the
schemas the model sees are unchanged — the guard is invisible from the prompt's
side. Read-only tools pass through untouched.

**The goal is at-least-once with deduplication, not exactly-once.** A crash
between the tool's own commit and the dedup record still repeats the call once.
That window is stated in the docstring rather than papered over; exactly-once
across a process boundary and a third-party API is not on offer.

### What this says about Waku

Waku's own tools do not close this gap, and the shape of the gap is the
interesting part. `waku/tools/calendar.py` guards `create_event` by natural key
(title + start) — but that guard runs *before* `conn.commit()`, the `.ics`
write and the Apple and Google syncs. So a replay is told
`"already exists (not duplicated)"` and **those syncs never run**. A replay
cannot distinguish "nothing to do" from "the downstream syncs were skipped".
`send_message` and `save_note` guard nothing at all.

That is a `waku/` bug on its own terms, independent of LangGraph, and
`tests/test_resume.py` pins the behaviour so the claim is not folklore.

## 8. Replanning

```python
make_replanner(propose)   # propose(goal, remaining, done, lessons) -> list[Step]
```

Roll back the work, keep the reflection. The replanner is handed the goal, the
**unfinished** steps, the finished results and the lessons, and returns the
steps that replace the unfinished ones. It returns a new `plan`, one
`Revision`, a cleared `replan_requested`, and an incremented `replans`.

Handing it the lessons is the point. A `Lesson` nothing reads is dead weight,
and the replanner is invoked at exactly the moment something went wrong — so
without them it can propose the failing approach again.

Two failure modes are handled explicitly:

- **A proposer that raises** leaves the plan alone (no `"plan"` key in the
  return, so the previous plan stands), records the error, and clears the
  request. Leaving the request set routes `replan ─► step ─► replan` for ever;
  a graph wired that way hits `GraphRecursionError`, which is how the guard was
  verified.
- **A proposal of nothing** leaves `plan == []`, and `after_plan` ends the
  task. That is a legitimate answer: there is nothing sensible left to do.

Every replan increments `replans`, a raising one included, because a replanner
that fails every time is exactly what the cap exists to stop.

## 9. Sessions, resume and the two databases

Two files, two jobs, and the split matters:

| File | Holds | Lifetime |
|---|---|---|
| the checkpointer's SQLite file | the task's progress | may be deleted with the task |
| Waku's `state.db` | the dedup table | must outlive any single task |

The dedup table is the record of what was already done to the world, so it
cannot live in a file that is thrown away with the task.

`open_saver` makes the connection itself and hands it back for the caller to
close, because `SqliteSaver.from_conn_string` is a context manager — fine for a
short block, wrong for a long-lived run.

`thread_config` pins `recursion_limit` to **200**. LangGraph's resolved default
is 10,007, which is not a guard at all; a planned task needing more than a
couple of hundred supersteps has gone wrong, and stopping is the right answer.

### How resume is actually proven

Five tests use a second connection in the same interpreter — close to a
restart, cheap enough to run every time. The README's claim is stronger than
that, so one test earns it properly:

```python
# a child process commits a side effect, then:
os._exit(1)       # no exception, no unwinding, no checkpoint
```

A second child resumes the thread and asserts the external file holds exactly
one line per step. That is a `kill -9` or a closed laptop lid, not a tidy
exception. It costs two process spawns and is the only test that proves the
claim without a caveat.

The checkpoint-size test runs twelve steps and asserts the state's growth stays
within one digest per step plus the plan plus one rollup — about 5,000 chars
where a 4KB output per step carried in state would be 48,000.

## 10. Termination: four bounds, and why one is not enough

The budget looks like the termination bound. It is not. **A step whose
`run_loop` raises before its first model call costs nothing**, so
`budget_spent` never moves and the step is selected again indefinitely. Failing
can be free.

Four bounds, in the order they bite:

| Bound | Value | Stops |
|---|---|---|
| `MAX_STEP_ATTEMPTS` | 2 | one step retried for ever |
| `MAX_REPLANS` | 5 | the replan cycle, whatever the step ids look like |
| `budget_limit` | caller's | work that costs money |
| `recursion_limit` | 200 | everything else |

The per-step bound alone is evadable: a replanner that renames a failed step
resets its lesson count to zero. That is precisely why `replans` is a single
global counter with a bare annotation, incremented in the replan node rather
than in the router — it counts where a replan actually happened, and a rename
cannot touch it.

## 11. The hyphen problem

The lab playbook fixes the directory name as `planning-agent`, and Python
cannot import a hyphen. The fix is `_path.py`, which registers a
`lab.planning_agent` alias in `sys.modules` pointing at the directory.

It cannot be imported the normal way — that is the chicken and egg — so both
entry points load it **by file path**:

- `tests/conftest.py` is a nine-line loader, so pytest works.
- `run.py` does the same four lines, so the README's documented
  `python lab/planning-agent/run.py` works on a clean shell.

Putting the alias only in `conftest.py` leaves `run.py` broken, which is why it
is a module of its own rather than a few lines in the conftest.

## 12. Where it touches Waku

Four imports, all one-directional. Nothing under `waku/` or `evals/` imports
anything here, and `test_rulebook.py` enforces that.

| Import | Used for |
|---|---|
| `waku.loop.trim.shrink_seen` | the in-step trim passed to `run_loop` |
| `waku.tools.registry.ToolRegistry`, `Tool` | building the guarded registry |
| `waku.loop.agent.run_loop` | injected by the caller, never imported here |
| `waku.db`, `waku.tools.calendar` | one test, pinning the `create_event` replay |

No new default dependency. `langgraph` and `langgraph-checkpoint-sqlite` are
installed by the topic's own `pip install` line, as conventions §6 rule 4
permits; `pyproject.toml` and `uv.lock` are untouched. Each module that imports
the SDK carries its own install line and a dated version header, as rule 5
requires.

## 13. The test suite

86 tests, 1,188 lines, nine modules. Every module opens with
`pytest.importorskip("langgraph")`, so on a default install the whole topic
skips cleanly — verified by blocking `langgraph` from `sys.meta_path`: 9
skipped, no collection error.

| Module | Tests | Covers |
|---|---|---|
| `test_state.py` | 8 | both reducers through a real compiled graph; id-based selection; the shrink case; the attempt cap |
| `test_digest.py` | 6 | exact strings, no model, free |
| `test_offload.py` | 12 | round trip of a 10k body; search; the closed-set id guard (6 parametrised) |
| `test_idempotency.py` | 10 | the key; failures not replayed; the stale-schema refusal; the paired `['a','b','b','c']` demonstration |
| `test_graph.py` | 14 | all 14 routing cases; the replan edge not buying a step; the cycle terminating |
| `test_bridge.py` | 16 | artifact parsing; the prompt's contents; the rollup filter; the usage meter; the guarded registry |
| `test_resume.py` | 6 | resume on a fresh connection; the real `os._exit(1)` kill; checkpoint size; the `create_event` replay |
| `test_replan.py` | 9 | work preserved; lessons passed; the raising proposer; the replan count |
| `test_rollup_flow.py` | 5 | the threshold; no double rollup; the measured ratios |

Every module was written **red first**, and nine guards were verified by
breaking them deliberately:

| Break | What goes red |
|---|---|
| drop `Annotated[..., operator.add]` on `done` | `['s2'] != ['s1','s2']` |
| loosen `STEP_ID` to `.+` | 7 tests, `../escape` among them |
| make `after_plan` return `"step"` always | 5 tests, zero and negative budget included |
| empty `READ_ONLY` | the read-only pass-through, and the closed-set test |
| drop `replan_requested: False` from the raise path | `KeyError`, and a graph that loops to `GraphRecursionError` |

## 14. Measurements

Run with stub nodes, so these are properties of the structure rather than of a
provider on the day. `python lab/planning-agent/run.py --measure 24`:

| Step | No rollups | With rollups |
|---|---|---|
| s0 | 905 | 905 |
| s7 | 5,587 | 5,587 |
| s8 | 6,252 | 1,841 |
| s15 | 10,913 | 6,502 |
| s16 | 11,579 | 2,757 |
| s23 | **16,241** | **7,419** |

The rollups column steps down at s8 and again at s16, once per eight finished
steps, and climbs linearly in between. It does not reset to zero: each rollup
stays in the prompt.

**2.19x** smaller at the last prompt. Growth over the run falls from 17.9x to
8.2x — but it stays **linear**: 3,574 chars at 12 steps against 7,419 at 24, a
factor of 2.08. One rollup accumulates per eight steps, so flat growth would
need a second compaction tier, a rollup of rollups, which this topic did not
build: there is no measurement past 24 steps to justify compounding the
summarising loss.

Two rules keep `measure` honest, both learned the hard way:

- It reads `STEP_DIGEST_CHARS` from `digest.py` rather than hardcoding a size.
  An earlier version measured at 120 chars against an 800-char rollup cap,
  which makes eight digests and one rollup nearly the same size — the mechanism
  looked worthless when it was the measurement that was wrong.
- It only ever **appends** to `done`, exactly as the reducer requires. An
  earlier version deleted rolled-up entries, which no graph can do, so it
  measured a prompt the real one never produced.

## 15. What is deliberately absent

Four things a reader might expect, each left out on purpose:

1. **LangGraph's store.** The spec listed it for facts that outlive the task,
   then said durable facts go through `waku/memory/consolidation.py` and
   nowhere else. Both cannot be true, and Waku's memory already exists. A
   second durable store beside `state.db` would be two answers to one question.
2. **Writing a finished task back to Waku's memory.** The spec asks for one
   episode per finished task but does not pin what it says or when it is
   written. It needs a design decision before it needs a task.
3. **The strategy split and the `decompose` node.** The spec routed on
   `state["strategy"]` across three strategies. They differ by prompt, not by
   topology, so it collapsed into one plan-execute-replan cycle. `strategy` is
   not a field, and a test asserts `"strategy" not in state` so it does not
   drift back in.
4. **A step reading its own earlier outputs.** `READ_ONLY` names
   `read_step_output` and `search_step_outputs`, but nothing registers them.
   `Scratch.read` and `Scratch.search` exist and are tested; wiring them up as
   tools is a follow-up.

## 16. Known limits

Stated here rather than left for someone to discover:

- **The budget gate sits after `plan`.** A limit of zero stops every step but
  still pays for the one planner call.
- **A rollup caps its summary at 800 chars** while the prompt builder drops
  every digest it covers, so the tail of a long phase can leave the prompt
  entirely — and with item 4 above unbuilt, no tool can read it back.
- **`looks_like_failure` is a prefix heuristic.** It matches the two shapes
  `ToolRegistry` and the tools actually produce. A tool returning its own error
  prose in some other shape is recorded `ok` and replayed. The status column
  narrows this; it does not close it.
- **`guarded_registry`'s wrapper takes `_tool` as a keyword**, so a tool with a
  parameter of that name would collide, and the underscore filter means two
  calls differing only in an underscore argument dedupe together.
- **Lab tests do not run in CI.** By design: conventions §6 rule 3 keeps
  `make gate` independent of `lab/`, because a third-party SDK's breaking
  release is their problem, not a red build. The rulebook tests still check the
  topic's playbook conformance, the one-way dependency and the sdist exclusion.

## 17. File map

```
lab/planning-agent/
├── README.md              121  the question, findings, verdict
├── write-up.md                 this file
├── _path.py                35  the lab.planning_agent alias, by file path
├── state.py               115  records, reducers, next_step, the two caps
├── digest.py               31  step_digest, make_rollup
├── offload.py              59  Scratch: write, read, search, the id guard
├── idempotency.py         180  READ_ONLY, call_key, DedupTable, guarded_registry
├── graph.py               114  after_plan, after_step, build_graph, maybe_roll_up
├── bridge.py              184  prompt, artifacts, UsageMeter, make_execute_step
├── session.py              54  open_saver, open_dedup, thread_config
├── replan.py               66  make_replanner, preserved
├── run.py                 119  measure, --measure, --smoke
└── tests/                1188  86 tests across 9 modules
```

Twelve commits, one per plan task plus the review pass:

```
58a0a22  Add the planning-agent lab topic with its playbook
4c0c345  Add PlanState: plan replaced, histories appended
b316e65  Add step digests and phase rollups
a166a97  Add the scratch directory for full step outputs
82d8e03  Make a resumed step safe to re-run
9f4401b  Add the plan-execute-replan topology with code routers
adad5a7  Run each planned step as one Waku loop turn
457c286  Resume a crashed task in a fresh process
8a2c8f6  Revise the plan from the failure point, keeping the lessons
655602c  Collapse finished phases to cut the prompt by about 2.2x
ce89540  Record the planning topic's findings and graduation verdict
819c566  Act on the two-axis review of the planning topic
```

The verdict this all feeds is in the README: `waku/graph/` should copy the idea
and leave the dependency — a `thread_id`, a state blob, a step cursor and two
reducers, at rung 1 of the footprint ladder.
