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
   in-memory checkpointing only, and phase 3 — the kill-and-resume check this
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
  digest.py          # step digests (reuses waku.memory.reports.digest shape)
  bridge.py          # execute_step -> waku.loop.agent.run_loop
  run.py             # one command, per the playbook's "Run it"
```

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
    budget_spent: float
    budget_limit: float
```

Two reducers, on purpose:

- **`plan` is overwritten.** It is always the current truth, short enough to
  recite into every step's prompt. One line per step, not a document.
- **`done` and `revisions` are appended.** They are the task's event history:
  what finished, and why the plan changed. A thirty-step task stays explicable
  after the fact instead of being a mystery.
- **`goal` never changes.** It is the guard against goal drift, re-injected on
  every step so a long task cannot wander off the original ask.

`Step` carries a status (`pending`, `in_progress`, `done`, `failed`).
`StepResult` carries the step id, a digest, and a pointer to the full output.
`Revision` carries what changed, and the reason a step was added or dropped.

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

## 6. Memory: three tiers

Waku already compresses within one turn: `waku/loop/trim.py:shrink_seen` cuts
long tool results the model has read, and `waku/memory/reports.py:digest` cuts
a report to its title, summary bullets and key numbers. Both act on the
`messages` list of **one** `run_loop` call. The measured reason they exist, from
`trim.py`: one research turn "sent 626.5k tokens in for 8.4k out: the model cost
$1.334."

Each `execute_step` is a separate `run_loop` call with its own fresh `messages`,
so `shrink_seen` protects the inside of a step and nothing protects the span
across steps. That gap is the middle tier below.

| Tier | Mechanism | Scope | Holds |
|---|---|---|---|
| Step working memory | `run_loop` messages + `shrink_seen` | one step | that step's transcript, discarded after |
| Task working memory | `plan`, `done` digests, `revisions` in `PlanState` | one task | one digest per completed step |
| Durable | LangGraph checkpointer; LangGraph store | thread; cross-thread | resume point; facts that outlive the task |

**What the next step sees.** The prompt for step N is: `goal`, the current
`plan` with statuses, and the **digests** in `done` — never raw transcripts.
This is what keeps a thirty-step task inside a context window.

**Drill-down, so the digest is not a one-way loss.** Full step outputs stay
addressable in the checkpointer, and `read_step_output(n)` is registered as a
tool in each step's scoped registry, so the model pulls a whole earlier output
on demand rather than the graph deciding for it. Summary by default, full text
on request. Without this, a detail dropped from step 3's digest silently breaks
step 9.

**Checkpointer and store are different things**, and keeping them distinct is
half the lesson: the checkpointer is this task's progress, scoped to a
`thread_id`, and makes a killed task resumable; the store holds facts that
outlive the task, across threads.

**What a finished task writes back to Waku.** One episode for the whole task,
and facts only through the existing `waku/memory/consolidation.py`. A thirty-step
task must not write thirty episodes into `state.db`, and this topic does not
reimplement consolidation.

## 7. Budget

`budget_spent` is in USD, the same unit as the `usage.jsonl` spend ledger, and
accumulates per step; a code gate checks it before each step.
On exhaustion the graph routes to `synthesize`, which answers from what the
task already gathered — the same shape as the loop's guardrail 2, which spends
one tools-off call rather than throwing the turn away.

The digest tier is the real cost control. `budget_limit` only decides when to
stop; digests decide how expensive each step is.

## 8. The bridge to the loop

`execute_step` calls `waku.loop.agent.run_loop` for one sub-task, the same move
`waku/graph/nodes.agent_node` makes: a scoped tool registry, its own working
memory, and the step's observer events passed through for tracing. `run_loop`
is not modified. This is what makes the topic a Waku experiment rather than a
generic LangGraph tutorial, and it is what the graduation assessment will judge.

## 9. How this gets checked

`lab/` is outside `make gate` by rule, so checks run from `run.py` and are
reported in the README's "What we found":

- **Resume is real.** Start a multi-step task, kill the process mid-step,
  restart, and assert it continues from the last completed step rather than
  restarting the task. This is the claim the whole topic rests on.
- **The update book holds.** Force a step to fail, assert the replanner
  preserves `done`, appends a `Revision`, and restructures only the remaining
  work.
- **Digests bound the context.** Run a task of at least ten steps and record
  tokens per step. The number to beat is flat growth; naive transcript passing
  grows quadratically.
- **Routers stay code.** No LLM call decides an edge.

A behaviour that graduates into `waku/` brings a deterministic eval with it at
that point, per hard rule 4. Nothing in `lab/` is exempt from that on the way in.

## 10. Phasing

Each phase is one component plan, written when it is reached.

| phase | deliverable | proves |
|---|---|---|
| 1 | `state.py` + `graph.py` with stub nodes, no LLM | the topology and the two reducers, at zero spend |
| 2 | `bridge.py`: `execute_step` calls `run_loop` | a real Waku turn as a LangGraph node |
| 3 | checkpointer + kill/resume check | the capability `waku/graph/` lacks |
| 4 | replanner + `revisions` | the plan survives discovered work |
| 5 | digests + `read_step_output` + token measurements | long tasks stay affordable |
| 6 | README "What we found" + "Graduation" | the verdict, with numbers |

## 11. Decisions taken

1. **LangGraph in `lab/` only.** The core engine's no-LangGraph decision stands.
2. **`plan` replaced, `done` and `revisions` appended.** Two reducers, one state.
3. **`goal` is immutable** and recited every step.
4. **Routers are code**, matching `waku/graph/` decision §7.4.
5. **Replanning is triggered** by failure or discovered work, not scheduled per step.
6. **`run_loop` is untouched**; `execute_step` composes it.
7. **Three memory tiers**, with step digests as the span-crossing tier.
8. **One episode per finished task**, facts via existing consolidation.
9. **The lab folder is created at implementation time**, with a verified version.
10. **The topic declares its own `pip install`** rather than amending the `arena`
    extra, which does not carry the SQLite checkpointer and is scoped to memory
    backends.

## 12. Open questions

1. **Does the checkpointer earn a place in `waku/graph/`?** The graduation
   question. Phase 3 produces the evidence; the answer may be "copy the idea,
   not the dependency" — a SQLite checkpointer for Waku's own engine is a small
   file, and `state.db` already exists.
2. **Where planning state would live after graduation.** LangGraph's
   checkpointer wants its own SQLite file; Waku keeps one `state.db`. If this
   graduates, the decision is whether plan progress becomes a table there.
3. **Whether the strategy split survives contact.** `deterministic`,
   `hierarchical` and `orchestrated` may collapse into one plan-execute-replan
   cycle with different prompts. Phase 1 should be built so that collapsing them
   is cheap.
