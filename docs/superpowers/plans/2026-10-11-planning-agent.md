# Planning Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph planning agent in `lab/planning-agent/` that decomposes a goal into steps, runs each step as a Waku `run_loop` turn, revises the plan when work is discovered, and resumes after a crash without repeating side effects.

**Architecture:** A `StateGraph` over `PlanState` with four nodes (`plan`, `execute_step`, `replan`, `synthesize`) and code-only routers. `execute_step` calls `waku.loop.agent.run_loop` for one sub-task, writes its full output to a per-task scratch directory, and merges back a short digest plus verbatim artifacts. A SQLite checkpointer makes the task resumable; a dedup table in `state.db` makes resume safe for side-effecting tools.

**Tech Stack:** Python 3.11+, langgraph 1.2.14, langgraph-checkpoint-sqlite 3.1.1, stdlib `sqlite3`, pytest 9.1.1. Waku's own `run_loop`, `ToolRegistry` and `waku.loop.trim`.

**Spec:** `docs/superpowers/specs/2026-10-11-planning-agent-design.md`

## Global Constraints

- **Nothing under `waku/` or `evals/` imports from `lab/`.** The dependency runs one way. `evals/deterministic/test_rulebook.py::test_product_and_evals_never_reach_into_examples_or_lab` enforces it.
- **`make gate` never depends on this folder.** `make eval` runs `pytest -q evals/deterministic` and `waku/ops/release_gate.py` runs `pytest -q <repo>/evals/<suite>`; both are path-scoped, so lab tests never enter the gate. Keep it that way: no lab test goes in `evals/`.
- **Every lab test module starts with `pytest.importorskip("langgraph")`.** There is no `[tool.pytest]` config in `pyproject.toml`, so a bare `pytest` from the repo root discovers `lab/`. The skip keeps a default install green.
- **No change to `pyproject.toml` and no new default dependency.** The topic states its own `pip install` in its README, which conventions §6 rule 4 permits. The `arena` extra is insufficient: it carries `langgraph` but not `langgraph-checkpoint-sqlite`.
- **Install line:** `pip install langgraph langgraph-checkpoint-sqlite`
- **`lab/planning-agent/README.md` must exist from the first commit**, carrying all six playbook headings — `The question`, `What we connect`, `Run it`, `What we found`, `Video angle`, `Graduation` — and a line matching `Verified against: .+, \d{4}-\d{2}-\d{2}`. `test_lab_topics_follow_the_playbook` fails the moment the directory exists without them.
- **No emojis** anywhere: docs prose, CLI output, code comments. `test_no_emoji_in_rulebook_or_readme` covers the rulebook; the rule is repo-wide.
- **No `.excalidraw` files** in the topic. Commit PNGs under `screenshots/` only.
- **Routers are code, never LLM calls.** Models write state; functions read it and pick the edge.
- **Hard rule 1: never wipe runtime data.** The dedup table lives in `state.db`. Nothing in this topic deletes or recreates it. The scratch directory is per task and may be removed freely.
- **Verified facts to build on (do not re-derive):** `SqliteSaver(conn)` takes a `sqlite3.Connection` and needs `.setup()`; `SqliteSaver.from_conn_string(s)` is a context manager. `invoke(None, config)` resumes a thread. `Annotated[list[X], operator.add]` appends; a bare annotation replaces. The default `recursion_limit` resolves to 10007. **Resume re-runs the whole failed node**, so a side effect that committed before the exception happens twice — this is measured, not theoretical (see Task 5). `ToolRegistry.execute` catches a tool's exception and returns `f"Error running {name}: {exc}"` as an ordinary string, so a failure is not distinguishable from a result by type. `Tool` is a dataclass, so `dataclasses.replace(tool, fn=...)` is how a guard is attached, and `wants_notify` carries over with it. `run_loop`'s observer is called as `notify(kind, event)` and its `llm` events carry `{"usage": {"in": N, "out": N}}`. `waku/tools/calendar.py` calls `conn.commit()` on line 441 and `_write_ics` on line 442, in that order, which is the skipped-sync hole. A rollup of eight 600-char digests against an 800-char cap saves about 2.2x on a 24-step prompt; the same measurement at a 120-char digest size saves almost nothing, which is why `measure` reads its size from `digest.py`.

## Deliberately not in this plan

Two things the spec mentions that no task below implements. Neither is an
oversight, and an executor should not add them unprompted.

1. **LangGraph's store is unused.** The spec's section 6 table lists it for
   facts that outlive the task, but section 6.5 then says durable facts go
   through `waku/memory/consolidation.py` and nowhere else. Both cannot be
   true, and Waku's memory is the one that already exists. The store stays out,
   and the README's findings should say so: a second durable store beside
   `state.db` would be two answers to one question.
2. **Writing a finished task back to Waku's memory.** Section 6.5 asks for one
   episode per finished task. The spec does not pin what that episode says or
   when it is written, and the phasing does not include it either. It needs a
   design decision before it needs a task, so it is left for a follow-up rather
   than guessed at here.
3. **The strategy split and the `decompose` node.** Spec section 5 routes on
   `state["strategy"]` across `deterministic`, `hierarchical` and
   `orchestrated`, with a `decompose` node for the middle one. It collapsed
   into one plan-execute-replan cycle, which spec section 13 question 3
   anticipated: the three strategies differ by prompt, not by topology. So
   `strategy` is not a field on `PlanState` and there is no strategy router.
   Adding either back is a design change, not a gap to fill in.

## Review Focus

1. **A replan that shrinks or reorders the plan** must not re-run a finished step or index past the end. Selecting the next step by `len(done)` breaks here; select by "first step whose id is not in `done`".
2. **An empty plan** (the planner returns no steps) must route straight to `synthesize` and answer, not spin the cycle or crash.
3. **A dedup key must include `thread_id`.** Without it, two different tasks performing the same action collide, and the second one is silently skipped — a real booking that never happens.
4. **A step id used in a scratch filename** must be validated against a closed character set. An id containing `/` or `..` writes outside the scratch directory.
5. **`budget_limit` of zero or less** must halt before any step runs, not after the first one has already been paid for.
6. **The replan edge must not buy a step.** `after_step` answers `replan` before it checks the budget, so a plain `replan -> execute_step` edge spends one more step past the limit. The edge is conditional, through `after_plan`, which already checks budget, an empty plan and nothing-left.
7. **A task must terminate even when nothing costs anything.** A `run_loop` that raises before its first model call leaves `budget_spent` untouched, so the budget is not a termination bound. Two bounds are: a step with two lessons against it is skipped, and a global `replans` count caps the cycle at five. The per-step bound alone is evadable, because a replanner that renames a failed step resets its lesson count.
8. **A dedup row must distinguish success from failure.** `ToolRegistry.execute` returns `f"Error running {name}: {exc}"` as an ordinary string, and tool bodies return their own error prose in no fixed format. A table that records results without a status replays a transient failure for ever.

---

### Task 1: Create the topic, pin the versions, keep CI green

**Files:**
- Create: `lab/planning-agent/README.md`
- Create: `lab/planning-agent/__init__.py`
- Create: `lab/planning-agent/tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the topic directory. Every later task adds files under it.

This task exists on its own because the directory and the playbook README are
inseparable: creating one without the other turns `test_rulebook.py` red.

- [ ] **Step 1: Install the two packages and record the exact versions**

```bash
python -m pip install langgraph langgraph-checkpoint-sqlite
python -c "import importlib.metadata as m; print('langgraph', m.version('langgraph')); print('langgraph-checkpoint-sqlite', m.version('langgraph-checkpoint-sqlite'))"
```

Expected: `langgraph 1.2.14` and `langgraph-checkpoint-sqlite 3.1.1` or newer.
If the numbers differ, use the numbers you actually saw in the README line
below — the point of that line is that it is true.

- [ ] **Step 2: Create the directory and the playbook README**

```bash
mkdir -p lab/planning-agent/tests
touch lab/planning-agent/__init__.py lab/planning-agent/tests/__init__.py
```

Write `lab/planning-agent/README.md`:

```markdown
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

Verified against: langgraph 1.2.14, 2026-10-11

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
```

- [ ] **Step 3: Run the rulebook tests to prove the topic is well-formed**

Run: `python -m pytest evals/deterministic/test_rulebook.py -q`
Expected: PASS, 17 tests. If `test_lab_topics_follow_the_playbook` fails, a
playbook heading is missing or the `Verified against:` line does not match
`Verified against: .+, \d{4}-\d{2}-\d{2}`.

- [ ] **Step 4: Prove a bare pytest run still works on a default install**

Run: `python -m pytest lab/planning-agent -q`
Expected: "no tests ran" — and specifically no collection error.

- [ ] **Step 5: Commit**

```bash
git add lab/planning-agent/
git commit -m "Add the planning-agent lab topic with its playbook"
```

---

### Task 2: `state.py` — the records, the reducers, and next-step selection

**Files:**
- Create: `lab/planning-agent/state.py`
- Create: `lab/planning-agent/_path.py`
- Test: `lab/planning-agent/tests/test_state.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `PlanState` (TypedDict), the record types `Step`, `StepResult`, `Revision`, `Lesson`, `Rollup`, `next_step(state) -> Step | None`, `new_state(goal, budget_limit) -> PlanState`, `MAX_STEP_ATTEMPTS`, `MAX_REPLANS`.
- Also produces `_path.py`, the import shim. The directory is `planning-agent` with a hyphen, which Python cannot import, so one module installs the `lab.planning_agent` alias and everything else imports it first. `tests/conftest.py` imports it so pytest works, and `run.py` imports it so the documented `python lab/planning-agent/run.py` command works. Putting the alias only in `conftest.py` leaves `run.py` broken on a clean shell.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_state.py`:

```python
import operator

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from lab.planning_agent.state import (
    MAX_REPLANS, MAX_STEP_ATTEMPTS, PlanState, attempts, new_state, next_step,
)


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def test_plan_is_replaced_and_done_is_appended():
    """The two reducers are the whole point: plan overwrites, done accumulates."""
    graph = StateGraph(PlanState)

    def first(state):
        return {"plan": [{"id": "s1", "description": "one", "status": "pending"}],
                "done": [{"step_id": "s1", "digest": "d1", "artifacts": {}, "output_path": "p1"}]}

    def second(state):
        return {"plan": [{"id": "s2", "description": "two", "status": "pending"}],
                "done": [{"step_id": "s2", "digest": "d2", "artifacts": {}, "output_path": "p2"}]}

    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    app = graph.compile(checkpointer=InMemorySaver())

    out = app.invoke(new_state("g", budget_limit=1.0),
                     {"configurable": {"thread_id": "t"}})

    assert [s["id"] for s in out["plan"]] == ["s2"], "plan must be replaced"
    assert [r["step_id"] for r in out["done"]] == ["s1", "s2"], "done must append"


def test_next_step_skips_finished_steps_by_id():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "A", "status": "pending"},
                     {"id": "b", "description": "B", "status": "pending"}]
    state["done"] = [{"step_id": "a", "digest": "", "artifacts": {}, "output_path": ""}]
    assert next_step(state)["id"] == "b"


def test_next_step_survives_a_replan_that_shrinks_the_plan():
    """Review Focus 1. Two steps finished, then the replanner cut the plan to
    one. Selecting by len(done) would index past the end; selecting by id
    returns None, which is 'nothing left to do'."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "A", "status": "pending"}]
    state["done"] = [{"step_id": "a", "digest": "", "artifacts": {}, "output_path": ""},
                     {"step_id": "b", "digest": "", "artifacts": {}, "output_path": ""}]
    assert next_step(state) is None


def test_next_step_on_an_empty_plan_is_none():
    """Review Focus 2."""
    assert next_step(new_state("g", budget_limit=1.0)) is None


def test_new_state_starts_every_appended_list_empty():
    state = new_state("goal text", budget_limit=2.5)
    assert state["goal"] == "goal text"
    assert state["budget_limit"] == 2.5
    assert state["budget_spent"] == 0.0
    assert state["replans"] == 0
    assert "strategy" not in state, "the strategy split collapsed; see the preamble"
    for key in ("plan", "done", "revisions", "lessons", "rollups"):
        assert state[key] == [], key


def test_a_step_that_failed_twice_is_skipped_rather_than_retried_for_ever():
    """Review Focus 7. A step whose run_loop raises before its first model call
    costs nothing, so the budget never advances and the step is selected again
    until the recursion limit stops the task."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["lessons"] = [
        {"step_id": "a", "attempt": "x", "failure": "boom", "diagnosis": "d"},
        {"step_id": "a", "attempt": "x", "failure": "boom", "diagnosis": "d"},
    ]
    assert next_step(state)["id"] == "b", "'a' is exhausted and must be passed over"


def test_one_failure_is_not_enough_to_abandon_a_step():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["lessons"] = [{"step_id": "a", "attempt": "x", "failure": "boom",
                         "diagnosis": "d"}]
    assert next_step(state)["id"] == "a", "one retry is the point of a lesson"


def test_attempts_counts_lessons_for_one_step_only():
    state = new_state("g", budget_limit=1.0)
    state["lessons"] = [{"step_id": "a", "attempt": "", "failure": "", "diagnosis": ""},
                        {"step_id": "b", "attempt": "", "failure": "", "diagnosis": ""}]
    assert attempts(state, "a") == 1
    assert attempts(state, "zzz") == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent'`.

Note the import name: the directory is `lab/planning-agent` (hyphen) but Python
cannot import a hyphen. Fix it in the next step by running pytest from the repo
root with the package path spelled as a file path, which is what Step 4 does.
If the import still fails, the cause is the hyphen, and the resolution is in
Step 3's `conftest.py`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/state.py`:

```python
"""The task's state: one goal, a plan that is replaced, and histories that grow.

The reducers are the design. `plan` carries no reducer, so each write replaces
it outright: it is always the current truth, short enough to recite into every
step's prompt. `done`, `revisions`, `lessons` and `rollups` use operator.add,
so they accumulate: they are the task's event history, and nothing in them is
ever rewritten.

`goal` never changes. It is re-injected on every step, which is what stops a
long task drifting off the original ask.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Step(TypedDict):
    id: str
    description: str
    status: str  # pending | in_progress | done | failed


class StepResult(TypedDict):
    step_id: str
    digest: str               # the short form the next step reads
    artifacts: dict[str, str]  # identifiers, verbatim
    output_path: str          # the full output, on disk


class Revision(TypedDict):
    after_step: str
    change: str
    reason: str


class Lesson(TypedDict):
    step_id: str
    attempt: str
    failure: str
    diagnosis: str


class Rollup(TypedDict):
    step_ids: list[str]
    summary: str


class PlanState(TypedDict):
    goal: str
    plan: list[Step]
    done: Annotated[list[StepResult], operator.add]
    revisions: Annotated[list[Revision], operator.add]
    lessons: Annotated[list[Lesson], operator.add]
    rollups: Annotated[list[Rollup], operator.add]
    budget_spent: float  # USD, the same unit as the usage.jsonl ledger
    budget_limit: float
    replan_requested: bool
    last_error: str
    # Replans so far, across the whole task. A bare annotation, so each write
    # replaces it. This is the termination bound that cannot be evaded: the
    # per-step one below counts lessons by step id, and a replanner that
    # renames a failed step resets that count to zero.
    replans: int


# Two attempts at a step, then it is skipped rather than retried for ever.
MAX_STEP_ATTEMPTS = 2
# And five replans for the whole task, whatever the step ids look like.
MAX_REPLANS = 5


def new_state(goal: str, budget_limit: float) -> PlanState:
    return PlanState(goal=goal, plan=[], done=[], revisions=[], lessons=[],
                     rollups=[], budget_spent=0.0, budget_limit=budget_limit,
                     replan_requested=False, last_error="", replans=0)


def attempts(state: PlanState, step_id: str) -> int:
    """How many times a step has been tried and left a lesson behind."""
    return sum(1 for lesson in state["lessons"] if lesson["step_id"] == step_id)


def next_step(state: PlanState) -> Step | None:
    """The first step in `plan` that is neither finished nor exhausted, or None.

    Selected by id, never by position. A replanner may reorder the plan, drop
    steps from it or shorten it, and `len(done)` can legitimately exceed
    `len(plan)` afterwards; an index would then point past the end or at a step
    that already ran.

    A step with MAX_STEP_ATTEMPTS lessons against it is skipped. Without that,
    a step that fails before its first model call costs nothing, leaves
    `budget_spent` untouched, and is selected again for ever: the budget is not
    a termination bound when failing is free.
    """
    finished = {result["step_id"] for result in state["done"]}
    for step in state["plan"]:
        if step["id"] in finished:
            continue
        if attempts(state, step["id"]) >= MAX_STEP_ATTEMPTS:
            continue
        return step
    return None
```

Write `lab/planning-agent/_path.py`, the one place the hyphen is worked
around. `tests/conftest.py` loads it so pytest works, and `run.py` loads it so
the README's documented command works on a clean shell.

Both load it by file path rather than by import, because the hyphen blocks an
ordinary import of anything in this directory, `_path.py` included. That is the
whole chicken-and-egg problem: the fix cannot be imported the normal way.
`run.py` does it with the same four lines `conftest.py` uses, assigning the
module a plain name (`lab_planning_agent_path`) so the import below reads
clearly:

```python
"""Make `lab/planning-agent` importable as `lab.planning_agent`.

Python cannot import a directory with a hyphen, and the directory name is
fixed by the lab playbook. Registering an alias in sys.modules is the smallest
fix that keeps both the directory name and ordinary import statements.

This lives in its own module rather than in tests/conftest.py because pytest is
not the only entry point: run.py is a documented command in the README, and a
conftest is never loaded for it.
"""

import importlib.util
import sys
from pathlib import Path

TOPIC = Path(__file__).resolve().parent


def install_alias() -> None:
    if "lab.planning_agent" in sys.modules:
        return
    lab = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("lab", loader=None, is_package=True))
    lab.__path__ = [str(TOPIC.parent)]
    sys.modules.setdefault("lab", lab)

    spec = importlib.util.spec_from_file_location(
        "lab.planning_agent", TOPIC / "__init__.py",
        submodule_search_locations=[str(TOPIC)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab.planning_agent"] = module
    spec.loader.exec_module(module)


install_alias()
```

Write `lab/planning-agent/tests/conftest.py`, which is only a loader for it:

```python
"""Install the lab.planning_agent alias before any test imports it."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_planning_agent_path", Path(__file__).resolve().parent.parent / "_path.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_state.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Prove the reducer test can fail**

Temporarily change `done` in `state.py` to a bare `list[StepResult]` with no
`Annotated[..., operator.add]`. Re-run the test.
Expected: `test_plan_is_replaced_and_done_is_appended` FAILS with
`['s2'] != ['s1', 's2']`. Put the annotation back and confirm PASS again.
Record in the commit message what you broke.

- [ ] **Step 6: Commit**

```bash
git add lab/planning-agent/state.py lab/planning-agent/_path.py lab/planning-agent/tests/
git commit -m "Add PlanState: plan replaced, histories appended"
```

---

### Task 3: `digest.py` — short forms of a step and of a finished phase

**Files:**
- Create: `lab/planning-agent/digest.py`
- Test: `lab/planning-agent/tests/test_digest.py`

**Interfaces:**
- Consumes: `StepResult`, `Rollup` from `lab.planning_agent.state`.
- Produces: `step_digest(output: str, limit: int = 600) -> str` and `make_rollup(results: list[StepResult], limit: int = 800) -> Rollup`.

Both are deterministic and call no model, so they cost nothing and the tests
pin exact strings. A model-written summary is a later option, not this task.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_digest.py`:

```python
import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.digest import make_rollup, step_digest


def test_a_short_output_is_returned_whole():
    assert step_digest("a calendar event was created") == "a calendar event was created"


def test_whitespace_is_collapsed():
    assert step_digest("two   lines\n\nhere") == "two lines here"


def test_a_long_output_is_cut_and_says_how_long_it_was():
    digest = step_digest("x" * 5000, limit=100)
    assert digest.startswith("x" * 100)
    assert "5,000 chars total" in digest
    assert len(digest) < 200


def test_an_empty_output_digests_to_an_empty_string():
    assert step_digest("") == ""


def test_a_rollup_names_its_steps_and_joins_their_digests():
    results = [
        {"step_id": "s1", "digest": "found three venues", "artifacts": {}, "output_path": "p1"},
        {"step_id": "s2", "digest": "booked the first", "artifacts": {}, "output_path": "p2"},
    ]
    rollup = make_rollup(results)
    assert rollup["step_ids"] == ["s1", "s2"]
    assert "found three venues" in rollup["summary"]
    assert "booked the first" in rollup["summary"]


def test_a_rollup_of_nothing_is_empty_rather_than_an_error():
    rollup = make_rollup([])
    assert rollup["step_ids"] == []
    assert rollup["summary"] == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_digest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.digest'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/digest.py`:

```python
"""Short forms. A digest stands in for one step's output; a rollup stands in
for a finished phase.

Neither calls a model. The full output is on disk either way (offload.py), so
these only decide what the prompt carries, and a deterministic rule is one a
test can pin to an exact string.
"""

from __future__ import annotations

from lab.planning_agent.state import Rollup, StepResult

STEP_DIGEST_CHARS = 600
ROLLUP_CHARS = 800
NOTE = " [...{total:,} chars total; read the step output for the rest]"


def step_digest(output: str, limit: int = STEP_DIGEST_CHARS) -> str:
    """One short paragraph standing in for a whole step output."""
    text = " ".join(output.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + NOTE.format(total=len(output))


def make_rollup(results: list[StepResult], limit: int = ROLLUP_CHARS) -> Rollup:
    """Several finished steps as one entry, so their digests leave the prompt."""
    summary = " ".join(f"{r['step_id']}: {r['digest']}" for r in results if r["digest"])
    if len(summary) > limit:
        summary = summary[:limit].rstrip() + NOTE.format(total=len(summary))
    return Rollup(step_ids=[r["step_id"] for r in results], summary=summary)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_digest.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add lab/planning-agent/digest.py lab/planning-agent/tests/test_digest.py
git commit -m "Add step digests and phase rollups"
```

---

### Task 4: `offload.py` — the scratch directory, read and searchable

**Files:**
- Create: `lab/planning-agent/offload.py`
- Test: `lab/planning-agent/tests/test_offload.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Scratch(root: Path)` with `.write(step_id: str, output: str) -> str`, `.read(step_id: str) -> str`, `.search(pattern: str) -> list[tuple[str, str]]`; and `STEP_ID` (the compiled closed-set pattern).

Full outputs live here rather than inside the checkpoint. A checkpointer
serialises state on every step, so a transcript carried in state is re-written
at every step after the one that produced it.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_offload.py`:

```python
import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.offload import Scratch


def test_what_is_written_can_be_read_back_whole(tmp_path):
    scratch = Scratch(tmp_path)
    body = "line one\nline two\n" + "x" * 10000
    path = scratch.write("s1", body)
    assert scratch.read("s1") == body
    assert str(tmp_path) in path


def test_search_finds_the_step_that_holds_a_detail(tmp_path):
    scratch = Scratch(tmp_path)
    scratch.write("s1", "the venue is Alexandra Palace\nseats 10000")
    scratch.write("s2", "the caterer is Blue Olive")
    hits = scratch.search("caterer")
    assert hits == [("s2", "the caterer is Blue Olive")]


def test_search_returns_every_matching_line_with_its_step(tmp_path):
    scratch = Scratch(tmp_path)
    scratch.write("s1", "price 10\nprice 20")
    scratch.write("s2", "no match here")
    assert scratch.search("price") == [("s1", "price 10"), ("s1", "price 20")]


def test_search_is_case_insensitive(tmp_path):
    scratch = Scratch(tmp_path)
    scratch.write("s1", "The Venue")
    assert scratch.search("venue") == [("s1", "The Venue")]


def test_reading_a_step_that_was_never_written_is_an_explicit_error(tmp_path):
    with pytest.raises(KeyError, match="no output for step 'nope'"):
        Scratch(tmp_path).read("nope")


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "x" * 65, "a b", "a.b"])
def test_an_unsafe_step_id_is_refused(tmp_path, bad):
    """Review Focus 4. The guard is a closed set: letters, digits, hyphen and
    underscore are allowed and everything else is refused, so an id can never
    steer the write outside the scratch directory."""
    with pytest.raises(ValueError, match="step id"):
        Scratch(tmp_path).write(bad, "body")


def test_an_unsafe_step_id_is_refused_on_read_too(tmp_path):
    with pytest.raises(ValueError, match="step id"):
        Scratch(tmp_path).read("../escape")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_offload.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.offload'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/offload.py`:

```python
"""The scratch directory: full step outputs, on disk, readable and searchable.

State holds a digest and a path; the bytes live here. Two reasons, both
practical. A checkpointer serialises the whole state on every step, so a
transcript kept in state is re-serialised on every later step. And a digest is
a one-way loss unless the original is still reachable, which is what read and
search are for: an index-addressed read alone only helps a caller that already
knows which step holds the detail.
"""

from __future__ import annotations

import re
from pathlib import Path

# A closed set: these characters are allowed and every other one is refused.
# An id reaches a filename, so anything looser lets a '/' or a '..' steer the
# write out of the scratch directory.
STEP_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _checked(step_id: str) -> str:
    if not STEP_ID.fullmatch(step_id):
        raise ValueError(
            f"step id {step_id!r} is not allowed: use 1 to 64 characters from "
            "letters, digits, hyphen and underscore")
    return step_id


class Scratch:
    """One task's outputs, one file per step."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, step_id: str) -> Path:
        return self.root / f"{_checked(step_id)}.txt"

    def write(self, step_id: str, output: str) -> str:
        path = self._path(step_id)
        path.write_text(output, encoding="utf-8")
        return str(path)

    def read(self, step_id: str) -> str:
        path = self._path(step_id)
        if not path.exists():
            raise KeyError(f"no output for step {step_id!r}")
        return path.read_text(encoding="utf-8")

    def search(self, pattern: str) -> list[tuple[str, str]]:
        """Every line matching `pattern`, as (step_id, line), step by step."""
        needle = pattern.lower()
        hits: list[tuple[str, str]] = []
        for path in sorted(self.root.glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if needle in line.lower():
                    hits.append((path.stem, line))
        return hits
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_offload.py -q`
Expected: PASS, 12 tests (the parametrised case counts six).

- [ ] **Step 5: Prove the path guard can fail**

Temporarily change `STEP_ID` to `re.compile(r".+")` and re-run.
Expected: the `../escape` cases FAIL. Put it back, confirm PASS, and say in the
commit message what you broke.

- [ ] **Step 6: Commit**

```bash
git add lab/planning-agent/offload.py lab/planning-agent/tests/test_offload.py
git commit -m "Add the scratch directory for full step outputs"
```

---

### Task 5: `idempotency.py` — so resume cannot repeat a side effect

**Files:**
- Create: `lab/planning-agent/idempotency.py`
- Test: `lab/planning-agent/tests/test_idempotency.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `READ_ONLY: frozenset[str]`, `is_side_effecting(tool: str) -> bool`, `looks_like_failure(result: str) -> bool`, `call_key(thread_id: str, step_id: str, tool: str, args: dict) -> str`, and `DedupTable(conn: sqlite3.Connection)` with `.setup()`, `.recorded(key) -> str | None`, `.record(key, result, status)`, `.guard(thread_id, step_id, tool, args, fn) -> str`.

**This task comes before the checkpointer on purpose.** Resuming a crashed
graph re-runs the whole failed node, which is measured: a step that appended to
an external list and then raised produced `['a','b','b','c']` in that list
while the checkpointed state stayed clean at `['a','b','c']`. The first test
below is that reproduction.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_idempotency.py`:

```python
import operator
import sqlite3
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from lab.planning_agent.idempotency import (
    DedupTable, call_key, is_side_effecting, looks_like_failure,
)


def test_a_tool_that_is_not_named_read_only_is_treated_as_side_effecting():
    """The guard is a closed set: read-only tools are named, and anything
    unknown is assumed to change the world. Guessing the other way round means
    a new tool silently skips the dedup table."""
    assert is_side_effecting("create_event") is True
    assert is_side_effecting("send_message") is True
    assert is_side_effecting("a_tool_nobody_has_written_yet") is True
    assert is_side_effecting("list_events") is False


def test_the_key_changes_with_every_part_of_the_call():
    base = ("t1", "s1", "create_event", {"title": "lunch"})
    assert call_key(*base) == call_key("t1", "s1", "create_event", {"title": "lunch"})
    assert call_key(*base) != call_key("t2", "s1", "create_event", {"title": "lunch"})
    assert call_key(*base) != call_key("t1", "s2", "create_event", {"title": "lunch"})
    assert call_key(*base) != call_key("t1", "s1", "send_message", {"title": "lunch"})
    assert call_key(*base) != call_key("t1", "s1", "create_event", {"title": "dinner"})


def test_the_key_ignores_argument_order():
    assert call_key("t", "s", "x", {"a": 1, "b": 2}) == call_key("t", "s", "x", {"b": 2, "a": 1})


def test_two_tasks_doing_the_same_thing_do_not_dedupe_against_each_other():
    """Review Focus 3. Without thread_id in the key, the second task's real
    booking is silently skipped."""
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    calls = []

    def book():
        calls.append("booked")
        return "booked"

    table.guard("task-1", "s1", "create_event", {"title": "lunch"}, book)
    table.guard("task-2", "s1", "create_event", {"title": "lunch"}, book)
    assert calls == ["booked", "booked"], "two separate tasks must both act"


def test_a_failed_call_is_recorded_but_not_replayed():
    """Review Focus 8. A timeout is transient. Recording it keeps the crash
    window closed; replaying it would freeze one bad minute into the task."""
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    results = iter(["Error running create_event: timed out", "event 7 created"])

    def book():
        return next(results)

    first = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    second = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    assert first.startswith("Error running")
    assert second == "event 7 created", "the retry must reach the tool"
    rows = conn.execute("SELECT status FROM planning_tool_calls").fetchall()
    assert [r[0] for r in rows] == ["ok"], "the error row was replaced by the success"


def test_a_raised_call_is_recorded_as_an_error_and_re_raised():
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()

    def boom():
        raise RuntimeError("the API refused")

    with pytest.raises(RuntimeError, match="refused"):
        table.guard("t", "s1", "create_event", {"title": "lunch"}, boom)
    row = conn.execute("SELECT status, result FROM planning_tool_calls").fetchone()
    assert row[0] == "error"
    assert "refused" in row[1]


def test_setup_refuses_a_table_that_predates_the_status_column():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE planning_tool_calls (key TEXT PRIMARY KEY, result TEXT)")
    conn.commit()
    with pytest.raises(RuntimeError, match="status column"):
        DedupTable(conn).setup()


def test_a_replayed_call_returns_the_recorded_result_without_running_again():
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    calls = []

    def book():
        calls.append("ran")
        return "event 7 created"

    first = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    second = table.guard("t", "s1", "create_event", {"title": "lunch"}, book)
    assert first == second == "event 7 created"
    assert calls == ["ran"], "the second call must not reach the tool"


class _S(TypedDict):
    plan: list[str]
    done: Annotated[list[str], operator.add]
    fail_once: bool


def _build(saver, world, table):
    """A graph whose step commits a side effect and may then crash."""

    def plan(state):
        return {"plan": ["a", "b", "c"]}

    def step(state):
        finished = set(state["done"])
        nxt = next(s for s in state["plan"] if s not in finished)

        def act():
            world.append(nxt)
            return nxt

        if table is None:
            act()
        else:
            table.guard("t1", nxt, "create_event", {"name": nxt}, act)
        if state.get("fail_once") and nxt == "b":
            raise RuntimeError("crash after the side effect committed")
        return {"done": [nxt]}

    def more(state):
        return "step" if len(state["done"]) < len(state["plan"]) else "done"

    graph = StateGraph(_S)
    graph.add_node("plan", plan)
    graph.add_node("step", step)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "step")
    graph.add_conditional_edges("step", more, {"step": "step", "done": END})
    return graph.compile(checkpointer=saver)


def _run_to_completion(tmp_path, table):
    """Crash mid-step, then resume, and report what the world saw."""
    conn = sqlite3.connect(tmp_path / "ckpt.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    world: list[str] = []
    config = {"configurable": {"thread_id": "t1"}}
    app = _build(saver, world, table)
    with pytest.raises(RuntimeError):
        app.invoke({"plan": [], "done": [], "fail_once": True}, config)
    app.update_state(config, {"fail_once": False})  # the transient cause is gone
    out = app.invoke(None, config)
    conn.close()
    return world, out["done"]


def test_without_the_dedup_table_resume_repeats_the_side_effect(tmp_path):
    """The failure this module exists to prevent, demonstrated. The
    checkpointed state is clean; the outside world is not."""
    world, done = _run_to_completion(tmp_path, table=None)
    assert done == ["a", "b", "c"], "state stays consistent"
    assert world == ["a", "b", "b", "c"], "but 'b' was committed twice"


def test_with_the_dedup_table_resume_does_not_repeat_the_side_effect(tmp_path):
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    world, done = _run_to_completion(tmp_path, table=table)
    assert done == ["a", "b", "c"]
    assert world == ["a", "b", "c"], "'b' must be committed exactly once"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_idempotency.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.idempotency'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/idempotency.py`:

```python
"""What a resumed step may safely do again.

A checkpoint records what the program knew when it was written. It cannot
record what already happened in the world. Resuming re-runs the whole failed
node, so a step that committed a write and then raised commits it twice: with
no dedup table, a three-step run that crashed in step 'b' left ['a','b','b','c']
behind while its checkpointed state read ['a','b','c'].

Waku's own tools do not close this. calendar.py guards create_event by natural
key, but messages.py and notes.py guard nothing, and that guard runs before the
ics file and the Apple and Google syncs: a replay reports "already exists" and
skips them for good. So the planner keys its own calls and remembers the
outcome rather than trusting the tool.

The aim is at-least-once with deduplication. Exactly-once across a process
boundary and a third-party API is not on offer; a dedup window is.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable

# Named read-only tools. Everything absent from this set is treated as
# side-effecting, which is the safe direction: a tool added later is deduped
# until someone decides it needs no guard. Naming the dangerous ones instead
# would be a guess about an open set.
READ_ONLY: frozenset[str] = frozenset({
    "list_events",
    "search_web",
    "read_note",
    "search_notes",
    "search_memory",
    "read_step_output",
    "search_step_outputs",
})


def is_side_effecting(tool: str) -> bool:
    return tool not in READ_ONLY


# ToolRegistry.execute catches a tool's exception and returns this prefix as an
# ordinary string, so a caller cannot tell a failure from a result by type
# alone. Tool bodies also return their own error prose; the match is therefore
# a heuristic, and the status column is what makes it correctable rather than
# load-bearing.
FAILURE_PREFIXES = ("Error running ", "Error:")


def looks_like_failure(result: str) -> bool:
    return result.lstrip().startswith(FAILURE_PREFIXES)


def call_key(thread_id: str, step_id: str, tool: str, args: dict) -> str:
    """The identity of one intended call.

    thread_id is in the key on purpose. Without it, two separate tasks that
    legitimately perform the same action collide, and the second one's write is
    skipped — a booking that silently never happens.
    """
    payload = json.dumps(
        {"thread": thread_id, "step": step_id, "tool": tool, "args": args},
        sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DedupTable:
    """Outcomes of side-effecting calls, keyed by call_key.

    Lives in state.db in real use. Nothing here drops or recreates the table:
    it is runtime data, and hard rule 1 applies.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def setup(self) -> None:
        """Create the table, then check the columns actually present.

        CREATE TABLE IF NOT EXISTS is silent about a table that already exists
        with an older shape, and the queries below would then fail somewhere
        far away. Nothing here drops or rewrites the table: it is runtime data
        under hard rule 1, so a stale schema is reported, never repaired.
        """
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS planning_tool_calls ("
            " key TEXT PRIMARY KEY,"
            " result TEXT NOT NULL,"
            " status TEXT NOT NULL,"
            " recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        self.conn.commit()
        columns = {row[1] for row in
                   self.conn.execute("PRAGMA table_info(planning_tool_calls)")}
        missing = {"key", "result", "status"} - columns
        if missing:
            raise RuntimeError(
                f"planning_tool_calls is missing {sorted(missing)}: it predates the "
                "status column. Nothing here will alter it; migrate it by hand.")

    def recorded(self, key: str) -> str | None:
        """The recorded result for a call that succeeded, or None.

        A row with status 'error' is deliberately not a hit. The call is
        remembered so a crash cannot lose it, and it is re-run because a
        timeout or a rate limit is transient; replaying it would freeze one bad
        minute into the task for ever.
        """
        row = self.conn.execute(
            "SELECT result FROM planning_tool_calls WHERE key = ? AND status = 'ok'",
            (key,)).fetchone()
        return row[0] if row else None

    def record(self, key: str, result: str, status: str = "ok") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO planning_tool_calls (key, result, status) "
            "VALUES (?, ?, ?)", (key, result, status))
        self.conn.commit()

    def guard(self, thread_id: str, step_id: str, tool: str, args: dict,
              fn: Callable[[], str]) -> str:
        """Run `fn` once per identical call, and return what it returned.

        Every outcome is recorded before this returns, so the next attempt
        after a crash finds it; only a successful one is replayed. A crash
        between the tool's own commit and this record still repeats the call
        once: that window is why the goal is a dedup window rather than
        exactly-once.

        A raised exception is recorded as an error and re-raised. The step above
        turns it into a Lesson, and the call stays retryable.
        """
        key = call_key(thread_id, step_id, tool, args)
        seen = self.recorded(key)
        if seen is not None:
            return seen
        try:
            result = fn()
        except Exception as exc:
            self.record(key, f"{type(exc).__name__}: {exc}", status="error")
            raise
        self.record(key, result,
                    status="error" if looks_like_failure(result) else "ok")
        return result
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_idempotency.py -q`
Expected: PASS, 10 tests. In particular
`test_without_the_dedup_table_resume_repeats_the_side_effect` passes by
observing the duplicate, and the next test passes by its absence.

- [ ] **Step 5: Commit**

```bash
git add lab/planning-agent/idempotency.py lab/planning-agent/tests/test_idempotency.py
git commit -m "$(cat <<'MSG'
Make a resumed step safe to re-run

Resuming re-runs the whole failed node, so a step that commits a write and
then raises commits it twice. A three-step run that crashed in step b left
['a','b','b','c'] in the world while its checkpoint read ['a','b','c'].

Side-effecting calls are keyed by thread, step, tool and arguments, and the
outcome is recorded before the step returns. Read-only tools are named and
everything else is assumed to change the world.
MSG
)"
```

---

### Task 6: `graph.py` — the topology and the code routers

**Files:**
- Create: `lab/planning-agent/graph.py`
- Test: `lab/planning-agent/tests/test_graph.py`

**Interfaces:**
- Consumes: `PlanState`, `next_step`, `new_state` from `lab.planning_agent.state`.
- Produces: `after_plan(state) -> str`, `after_step(state) -> str`, and `build_graph(*, plan_fn, execute_fn, replan_fn, synthesize_fn, checkpointer=None)` returning a compiled graph.

The four node functions are injected, exactly as `waku/graph/workflows/triage.py`
injects its callables: tests script them, and the real client is bound later.
No node here calls a model, so this whole task runs at zero spend.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_graph.py`:

```python
import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from lab.planning_agent.graph import after_plan, after_step, build_graph
from lab.planning_agent.state import new_state

CONFIG = {"configurable": {"thread_id": "t"}}


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def _result(step_id):
    return {"step_id": step_id, "digest": step_id, "artifacts": {}, "output_path": ""}


def _plan_of(*ids):
    def plan_fn(state):
        return {"plan": _steps(*ids)}
    return plan_fn


def _executor(log):
    def execute_fn(state):
        from lab.planning_agent.state import next_step
        step = next_step(state)
        log.append(step["id"])
        return {"done": [_result(step["id"])], "budget_spent": state["budget_spent"] + 0.1}
    return execute_fn


def _noop(state):
    return {}


def _synth(state):
    return {"last_error": ""}


def test_an_empty_plan_routes_straight_to_synthesize():
    """Review Focus 2."""
    state = new_state("g", budget_limit=5.0)
    assert after_plan(state) == "synthesize"


def test_a_zero_budget_halts_before_any_step_runs():
    """Review Focus 5. The check sits after `plan`, so nothing is paid for."""
    state = new_state("g", budget_limit=0.0)
    state["plan"] = _steps("a")
    assert after_plan(state) == "synthesize"


def test_a_negative_budget_also_halts():
    state = new_state("g", budget_limit=-1.0)
    state["plan"] = _steps("a")
    assert after_plan(state) == "synthesize"


def test_a_plan_with_work_and_budget_routes_to_the_first_step():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a")
    assert after_plan(state) == "step"


def test_after_a_step_more_work_routes_back_to_the_step_node():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["done"] = [_result("a")]
    assert after_step(state) == "step"


def test_after_the_last_step_routes_to_synthesize():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a")
    state["done"] = [_result("a")]
    assert after_step(state) == "synthesize"


def test_a_requested_replan_wins_over_remaining_work():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["done"] = [_result("a")]
    state["replan_requested"] = True
    assert after_step(state) == "replan"


def test_an_exhausted_budget_stops_the_cycle_even_with_work_left():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["done"] = [_result("a")]
    state["budget_spent"] = 1.0
    assert after_step(state) == "synthesize"


def test_a_replan_is_refused_once_the_cap_is_reached():
    """Review Focus 7. Without this the cycle runs until the recursion limit,
    and every lap costs a model call once the nodes are real."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["replan_requested"] = True
    state["replans"] = 5
    assert after_step(state) == "step", "work is left, so carry on without replanning"

    state["done"] = [_result("a"), _result("b")]
    assert after_step(state) == "synthesize", "and answer when nothing is left"


def test_a_replan_under_the_cap_still_wins():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = _steps("a", "b")
    state["replan_requested"] = True
    state["replans"] = 4
    assert after_step(state) == "replan"


def test_the_replan_edge_does_not_buy_a_step_past_the_budget():
    """Review Focus 6. The replan node runs, and then the same budget gate
    that guards the first step guards the next one."""
    log: list[str] = []

    def failing_once(state):
        # One step that asks for a replan without spending anything, which is
        # what a provider refusal looks like.
        return {"replan_requested": True, "budget_spent": state["budget_limit"]}

    app = build_graph(plan_fn=_plan_of("a", "b"), execute_fn=failing_once,
                      replan_fn=lambda s: {"replan_requested": False,
                                           "replans": s["replans"] + 1},
                      synthesize_fn=_synth, checkpointer=InMemorySaver())
    out = app.invoke(new_state("g", budget_limit=1.0), CONFIG)
    assert out["done"] == [], "no step completed"
    assert out["replans"] == 1, "one replan ran, and then the budget stopped it"


def test_the_cycle_runs_every_step_once_and_stops():
    log: list[str] = []
    app = build_graph(plan_fn=_plan_of("a", "b", "c"), execute_fn=_executor(log),
                      replan_fn=_noop, synthesize_fn=_synth,
                      checkpointer=InMemorySaver())
    out = app.invoke(new_state("g", budget_limit=10.0), CONFIG)
    assert log == ["a", "b", "c"]
    assert [r["step_id"] for r in out["done"]] == ["a", "b", "c"]


def test_the_cycle_stops_when_the_budget_runs_out_midway():
    log: list[str] = []
    app = build_graph(plan_fn=_plan_of("a", "b", "c"), execute_fn=_executor(log),
                      replan_fn=_noop, synthesize_fn=_synth,
                      checkpointer=InMemorySaver())
    app.invoke(new_state("g", budget_limit=0.2), CONFIG)
    assert log == ["a", "b"], (
        "0.1 a step against a 0.2 limit stops after two: the guard is "
        "spent >= limit, so 0.2 >= 0.2 halts where 0.2 >= 0.25 would not")


def test_an_empty_plan_reaches_the_end_without_executing_anything():
    log: list[str] = []
    app = build_graph(plan_fn=lambda s: {"plan": []}, execute_fn=_executor(log),
                      replan_fn=_noop, synthesize_fn=_synth,
                      checkpointer=InMemorySaver())
    app.invoke(new_state("g", budget_limit=10.0), CONFIG)
    assert log == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.graph'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/graph.py`:

```python
"""The topology: plan, then a bounded execute cycle, then one answer.

    START -> plan -> after_plan -> execute_step -> after_step -> execute_step
                          |                            |  \\-> replan -> execute_step
                          \\-> synthesize -> END        \\-> synthesize -> END

Control flow is code. The nodes write state and these two router functions read
it, which is the rule `waku/graph/` already follows: models produce state,
functions pick the edge. Nothing here asks a model which way to go.

The node functions are injected so tests script them and the real client is
bound at the edge, the same shape as waku/graph/workflows/triage.py.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from lab.planning_agent.state import MAX_REPLANS, PlanState, next_step

NodeFn = Callable[[PlanState], dict]


def _out_of_budget(state: PlanState) -> bool:
    return state["budget_spent"] >= state["budget_limit"]


def after_plan(state: PlanState) -> str:
    """Where to go once a plan exists.

    Both checks happen before any step runs. A limit of zero must not buy one
    step, and an empty plan must answer rather than spin the cycle.
    """
    if state["budget_limit"] <= 0 or _out_of_budget(state):
        return "synthesize"
    if next_step(state) is None:
        return "synthesize"
    return "step"


def after_step(state: PlanState) -> str:
    """Where to go once a step has finished.

    A requested replan comes first: it is set by a step that failed or that
    discovered work, and in both cases the remaining plan is not trustworthy.
    It does not come before the replan cap, though. A step that fails before
    its first model call costs nothing, so the budget cannot stop that cycle
    and MAX_REPLANS is what does.
    """
    if state["replan_requested"] and state["replans"] < MAX_REPLANS:
        return "replan"
    if _out_of_budget(state):
        return "synthesize"
    if next_step(state) is None:
        return "synthesize"
    return "step"


def build_graph(*, plan_fn: NodeFn, execute_fn: NodeFn, replan_fn: NodeFn,
                synthesize_fn: NodeFn, checkpointer=None):
    graph = StateGraph(PlanState)
    graph.add_node("plan", plan_fn)
    graph.add_node("execute_step", execute_fn)
    graph.add_node("replan", replan_fn)
    graph.add_node("synthesize", synthesize_fn)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", after_plan, {"step": "execute_step", "synthesize": "synthesize"})
    graph.add_conditional_edges(
        "execute_step", after_step,
        {"step": "execute_step", "replan": "replan", "synthesize": "synthesize"})
    # Conditional, not a plain edge. after_step answers "replan" before it
    # checks the budget, so a plain replan -> execute_step edge buys one step
    # past the limit. after_plan already checks the budget, an empty plan and
    # nothing-left, which is exactly the question being asked here.
    graph.add_conditional_edges(
        "replan", after_plan, {"step": "execute_step", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_graph.py -q`
Expected: PASS, 14 tests.

- [ ] **Step 5: Prove the budget guard can fail**

Temporarily make `after_plan` return `"step"` unconditionally. Re-run.
Expected: `test_a_zero_budget_halts_before_any_step_runs` FAILS. Put it back,
confirm PASS, and name what you broke in the commit message.

- [ ] **Step 6: Commit**

```bash
git add lab/planning-agent/graph.py lab/planning-agent/tests/test_graph.py
git commit -m "Add the plan-execute-replan topology with code routers"
```

---

### Task 7: `bridge.py` — one step is one Waku loop turn

**Files:**
- Create: `lab/planning-agent/bridge.py`
- Test: `lab/planning-agent/tests/test_bridge.py`

**Interfaces:**
- Consumes: `PlanState`, `Step`, `next_step` from state; `Scratch` from offload; `step_digest` from digest.
- Produces: `ARTIFACTS_MARKER`, `parse_artifacts(reply: str) -> dict[str, str]`, `step_prompt(state: PlanState, step: Step) -> str`, `UsageMeter`, and `make_execute_step(*, run_loop, client, model, system, tools, scratch, thread_id, price_in, price_out, dedup=None) -> NodeFn`.
- Also adds to `lab/planning-agent/idempotency.py`: `guarded_registry(tools, table, thread_id, step_id) -> ToolRegistry`.

**Why the registry is wrapped here.** `run_loop` executes tools itself, through
`ToolRegistry.execute`. A dedup table the bridge merely holds would never be
consulted, so the guard has to sit on the tools the loop is given. Steps 5 to 9
add that; without them the dedup table built in Task 5 is unreachable from the
real path.

`run_loop` is a parameter rather than a direct import so the tests substitute a
fake and never touch a network. The real caller passes
`waku.loop.agent.run_loop`.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_bridge.py`:

```python
import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.bridge import (
    ARTIFACTS_MARKER, UsageMeter, make_execute_step, parse_artifacts, step_prompt,
)
from lab.planning_agent.offload import Scratch
from lab.planning_agent.state import new_state


class _FakeResult:
    def __init__(self, reply, tool_calls=None):
        self.reply = reply
        self.tool_calls = tool_calls or []
        self.iterations = 1
        self.limit_reached = False


def _fake_run_loop(reply, usage=(1000, 500)):
    """Stands in for waku.loop.agent.run_loop, including its observer calls."""
    def run_loop(*, client, model, system, messages, tools, max_iterations,
                 max_tokens, observer=None, trim=None, **kwargs):
        if observer:
            observer("llm", {"iteration": 1, "usage": {"in": usage[0], "out": usage[1]}})
        return _FakeResult(reply)
    return run_loop


def test_artifacts_are_parsed_from_the_marker_line():
    reply = f"Booked it.\n{ARTIFACTS_MARKER} event_id=abc123; url=https://x.test/1"
    assert parse_artifacts(reply) == {"event_id": "abc123", "url": "https://x.test/1"}


def test_a_reply_with_no_marker_has_no_artifacts():
    assert parse_artifacts("just prose") == {}


def test_an_identifier_is_kept_verbatim_rather_than_summarised():
    """The reason artifacts exist: a digest may paraphrase an id, and a later
    step needs it exactly."""
    reply = f"done\n{ARTIFACTS_MARKER} id=EV-2026-10-11-0001"
    assert parse_artifacts(reply)["id"] == "EV-2026-10-11-0001"


def test_a_malformed_artifact_pair_is_skipped_not_fatal():
    reply = f"x\n{ARTIFACTS_MARKER} good=1; nonsense; alsogood=2"
    assert parse_artifacts(reply) == {"good": "1", "alsogood": "2"}


def test_the_prompt_carries_the_goal_the_plan_and_earlier_digests():
    state = new_state("throw a party", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "find a venue", "status": "pending"},
                     {"id": "b", "description": "book it", "status": "pending"}]
    state["done"] = [{"step_id": "a", "digest": "three venues found",
                      "artifacts": {"shortlist": "v1,v2"}, "output_path": "p"}]
    prompt = step_prompt(state, state["plan"][1])

    assert "throw a party" in prompt
    assert "find a venue" in prompt and "book it" in prompt
    assert "three venues found" in prompt
    assert "shortlist=v1,v2" in prompt
    assert "book it" in prompt.split("Your step now")[1]


def test_the_prompt_carries_lessons_so_a_retry_does_not_repeat_them():
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "a", "description": "A", "status": "pending"}]
    state["lessons"] = [{"step_id": "a", "attempt": "called the venue",
                         "failure": "line was dead", "diagnosis": "use email"}]
    prompt = step_prompt(state, state["plan"][0])
    assert "line was dead" in prompt and "use email" in prompt


def test_the_prompt_carries_rollups_instead_of_the_digests_they_replaced():
    """The rollup has to REPLACE the digests, not join them. `done` is
    append-only, so nothing can leave it; this filter is the only place the
    collapse reaches the prompt. Without it a rollup makes the prompt longer."""
    state = new_state("g", budget_limit=1.0)
    state["plan"] = [{"id": "z", "description": "Z", "status": "pending"}]
    state["done"] = [
        {"step_id": "a", "digest": "venue A was free", "artifacts": {}, "output_path": ""},
        {"step_id": "b", "digest": "caterer B replied", "artifacts": {}, "output_path": ""},
        {"step_id": "c", "digest": "not yet rolled up", "artifacts": {}, "output_path": ""},
    ]
    state["rollups"] = [{"step_ids": ["a", "b"], "summary": "phase one settled the venue"}]
    prompt = step_prompt(state, state["plan"][0])

    assert "phase one settled the venue" in prompt
    assert "venue A was free" not in prompt, "a rolled-up digest must be gone"
    assert "caterer B replied" not in prompt
    assert "not yet rolled up" in prompt, "an unrolled digest stays"


def test_a_usage_meter_turns_token_counts_into_dollars():
    meter = UsageMeter(price_in=3.0, price_out=15.0)  # USD per million
    meter("llm", {"usage": {"in": 1_000_000, "out": 0}})
    assert meter.cost == pytest.approx(3.0)
    meter("llm", {"usage": {"in": 0, "out": 1_000_000}})
    assert meter.cost == pytest.approx(18.0)


def test_the_meter_ignores_events_that_are_not_llm_calls():
    meter = UsageMeter(price_in=3.0, price_out=15.0)
    meter("tool", {"tool": "x"})
    assert meter.cost == 0.0


def test_a_step_writes_its_full_output_and_merges_back_a_digest(tmp_path):
    state = new_state("g", budget_limit=5.0)
    state["plan"] = [{"id": "s1", "description": "do the thing", "status": "pending"}]
    body = "y" * 4000
    execute = make_execute_step(
        run_loop=_fake_run_loop(f"{body}\n{ARTIFACTS_MARKER} id=7"),
        client=object(), model="m", system="s", tools=object(),
        scratch=Scratch(tmp_path), thread_id="t", price_in=3.0, price_out=15.0)

    merged = execute(state)
    result = merged["done"][0]

    assert result["step_id"] == "s1"
    assert result["artifacts"] == {"id": "7"}
    assert len(result["digest"]) < 800, "the digest must be short"
    assert Scratch(tmp_path).read("s1").startswith("y" * 100), "the full output is on disk"
    assert merged["budget_spent"] > 0.0


def test_a_step_that_raises_records_a_lesson_and_asks_for_a_replan(tmp_path):
    state = new_state("g", budget_limit=5.0)
    state["plan"] = [{"id": "s1", "description": "do the thing", "status": "pending"}]

    def boom(**kwargs):
        raise RuntimeError("the provider refused")

    merged = make_execute_step(
        run_loop=boom, client=object(), model="m", system="s", tools=object(),
        scratch=Scratch(tmp_path), thread_id="t", price_in=3.0, price_out=15.0)(state)

    assert merged["replan_requested"] is True
    assert merged["lessons"][0]["step_id"] == "s1"
    assert "refused" in merged["lessons"][0]["failure"]
    assert merged["done"] == [], "a failed step produces no result"


def test_a_step_with_nothing_left_to_do_is_a_no_op(tmp_path):
    state = new_state("g", budget_limit=5.0)
    execute = make_execute_step(
        run_loop=_fake_run_loop("x"), client=object(), model="m", system="s",
        tools=object(), scratch=Scratch(tmp_path), thread_id="t",
        price_in=3.0, price_out=15.0)
    assert execute(state) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_bridge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.bridge'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/bridge.py`:

```python
"""One planned step is one Waku loop turn.

`run_loop` is not modified and not wrapped in anything clever: the step builds
a prompt, calls it with a scoped registry, and keeps what came back. This is
the same move waku/graph/nodes.agent_node makes, and it is what makes this a
Waku experiment rather than a LangGraph tutorial.

What the step reads is the point. It gets the goal, the plan, the rollups of
finished phases, the digests and artifacts of recent steps, and any lessons.
It never gets an earlier step's transcript: that is on disk, and the step can
ask for it.
"""

from __future__ import annotations

from collections.abc import Callable

from waku.loop.trim import shrink_seen

from lab.planning_agent.digest import step_digest
from lab.planning_agent.offload import Scratch
from lab.planning_agent.state import Lesson, PlanState, Step, StepResult, next_step

ARTIFACTS_MARKER = "ARTIFACTS:"

PROMPT = """\
You are working through one step of a larger task.

The overall goal, which never changes:
{goal}

The plan as it stands:
{plan}
{history}
Your step now: {step}

When your step produces identifiers that a later step will need exactly — an
event id, a file path, a URL — end your reply with one line:
{marker} key=value; key=value

Do not restate the plan. Report what you did and what you found."""


class UsageMeter:
    """Turns the loop's llm events into dollars.

    run_loop reports token counts per call through its observer; a receipt is
    assembled higher up, in Waku.respond, which a lab run does not go through.
    Prices are USD per million tokens, the unit providers quote.
    """

    def __init__(self, price_in: float, price_out: float) -> None:
        self.price_in = price_in
        self.price_out = price_out
        self.cost = 0.0

    def __call__(self, kind: str, event: dict) -> None:
        if kind != "llm":
            return
        usage = event.get("usage") or {}
        self.cost += (usage.get("in", 0) / 1_000_000) * self.price_in
        self.cost += (usage.get("out", 0) / 1_000_000) * self.price_out


def parse_artifacts(reply: str) -> dict[str, str]:
    """Identifiers from the reply's marker line, kept exactly as written.

    A pair without an '=' is skipped rather than fatal: the line is written by
    a model, so it is input, and a malformed half must not lose the good half.
    """
    for line in reversed(reply.splitlines()):
        if not line.strip().startswith(ARTIFACTS_MARKER):
            continue
        body = line.strip()[len(ARTIFACTS_MARKER):]
        found: dict[str, str] = {}
        for pair in body.split(";"):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            if key.strip():
                found[key.strip()] = value.strip()
        return found
    return {}


def _history(state: PlanState) -> str:
    """What has happened, with anything already summarised left out.

    A rolled-up step's digest must not also appear on its own, or the rollup
    adds to the prompt instead of shortening it. `done` is append-only, so a
    rollup cannot remove anything from it; filtering here is how the collapse
    actually reaches the prompt.
    """
    rolled_up = {sid for rollup in state["rollups"] for sid in rollup["step_ids"]}
    lines: list[str] = []
    for rollup in state["rollups"]:
        lines.append(f"- earlier phase ({', '.join(rollup['step_ids'])}): {rollup['summary']}")
    for result in state["done"]:
        if result["step_id"] in rolled_up:
            continue
        artifacts = "; ".join(f"{k}={v}" for k, v in result["artifacts"].items())
        lines.append(f"- {result['step_id']}: {result['digest']}"
                     + (f" [{artifacts}]" if artifacts else ""))
    for lesson in state["lessons"]:
        lines.append(f"- lesson from {lesson['step_id']}: tried {lesson['attempt']}, "
                     f"{lesson['failure']}. {lesson['diagnosis']}")
    if not lines:
        return ""
    return "\nWhat has happened so far:\n" + "\n".join(lines) + "\n"


def step_prompt(state: PlanState, step: Step) -> str:
    plan = "\n".join(
        f"{i}. [{s['status']}] {s['description']}" for i, s in enumerate(state["plan"], 1))
    return PROMPT.format(goal=state["goal"], plan=plan, history=_history(state),
                         step=step["description"], marker=ARTIFACTS_MARKER)


# `trim=shrink_seen` is passed to run_loop below. It is the first of the four
# memory mechanisms and the only one that already exists in waku: inside a
# single step, a tool result the model has read is cut before the next call.
# The other three mechanisms cover the span between steps, which this one
# cannot see.


def make_execute_step(*, run_loop: Callable, client, model: str, system: str,
                      tools, scratch: Scratch, thread_id: str,
                      price_in: float, price_out: float, dedup=None,
                      max_iterations: int = 10, max_tokens: int = 2048) -> Callable:
    """The execute_step node, bound to one client, registry and scratch."""

    def execute(state: PlanState) -> dict:
        step = next_step(state)
        if step is None:
            return {}

        scoped = tools
        if dedup is not None:
            from lab.planning_agent.idempotency import guarded_registry
            scoped = guarded_registry(tools, dedup, thread_id, step["id"])

        meter = UsageMeter(price_in=price_in, price_out=price_out)
        messages = [{"role": "user", "content": step_prompt(state, step)}]
        try:
            result = run_loop(client=client, model=model, system=system,
                              messages=messages, tools=scoped,
                              max_iterations=max_iterations, max_tokens=max_tokens,
                              observer=meter, trim=shrink_seen)
        except Exception as exc:
            # Surface, do not crash: the graph decides what happens next, and a
            # lesson is what stops the replanner proposing this again.
            return {"replan_requested": True,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "budget_spent": state["budget_spent"] + meter.cost,
                    "lessons": [Lesson(step_id=step["id"],
                                       attempt=step["description"],
                                       failure=f"{type(exc).__name__}: {exc}",
                                       diagnosis="the step did not complete; "
                                                 "try a different approach")]}

        output_path = scratch.write(step["id"], result.reply)
        return {"done": [StepResult(step_id=step["id"],
                                    digest=step_digest(result.reply),
                                    artifacts=parse_artifacts(result.reply),
                                    output_path=output_path)],
                "budget_spent": state["budget_spent"] + meter.cost,
                "replan_requested": False}

    return execute
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_bridge.py -q`
Expected: PASS, 12 tests. The rollup test is the one that matters: it is the
only check that a rollup shortens the prompt rather than adding to it.

- [ ] **Step 5: Write the failing test for the guarded registry**

Append to `lab/planning-agent/tests/test_bridge.py`:

```python
import sqlite3

from waku.tools.registry import Tool, ToolRegistry

from lab.planning_agent.idempotency import DedupTable, guarded_registry


def _registry(calls):
    registry = ToolRegistry()

    def create_event(title: str = "", start: str = "") -> str:
        calls.append(("create_event", title))
        return f"created {title}"

    def list_events(day: str = "") -> str:
        calls.append(("list_events", day))
        return "two events"

    for name, fn in (("create_event", create_event), ("list_events", list_events)):
        registry.register(Tool(name=name, description=name,
                               input_schema={"type": "object", "properties": {}}, fn=fn))
    return registry


def _table():
    conn = sqlite3.connect(":memory:")
    table = DedupTable(conn)
    table.setup()
    return table


def test_a_side_effecting_tool_runs_once_per_step_however_often_it_is_called():
    calls: list[tuple] = []
    guarded = guarded_registry(_registry(calls), _table(), "t1", "s1")

    first = guarded.execute("create_event", {"title": "lunch", "start": "x"})
    second = guarded.execute("create_event", {"title": "lunch", "start": "x"})

    assert first == second == "created lunch"
    assert calls == [("create_event", "lunch")], "the tool must run once"


def test_a_read_only_tool_is_not_deduped():
    # Re-reading is free and must stay free: a step that lists events twice
    # should see the second answer, not the first one replayed.
    calls: list[tuple] = []
    guarded = guarded_registry(_registry(calls), _table(), "t1", "s1")
    guarded.execute("list_events", {"day": "monday"})
    guarded.execute("list_events", {"day": "monday"})
    assert calls == [("list_events", "monday"), ("list_events", "monday")]


def test_the_same_call_in_a_later_step_is_a_different_call():
    calls: list[tuple] = []
    table = _table()
    registry = _registry(calls)
    guarded_registry(registry, table, "t1", "s1").execute("create_event", {"title": "a"})
    guarded_registry(registry, table, "t1", "s2").execute("create_event", {"title": "a"})
    assert len(calls) == 2, "step 2 intends a new action, not a replay"


def test_the_schemas_the_model_sees_are_unchanged():
    guarded = guarded_registry(_registry([]), _table(), "t1", "s1")
    names = sorted(s["name"] for s in guarded.schemas())
    assert names == ["create_event", "list_events"]
```

Run: `python -m pytest lab/planning-agent/tests/test_bridge.py -q`
Expected: FAIL — `ImportError: cannot import name 'guarded_registry'`.

- [ ] **Step 6: Add `guarded_registry` to `idempotency.py`**

Append to `lab/planning-agent/idempotency.py`:

```python
def guarded_registry(tools, table: DedupTable, thread_id: str, step_id: str):
    """The same tools, with the side-effecting ones deduped for this step.

    run_loop executes tools itself, so this is the only place a guard can sit
    and still be reached. Read-only tools pass through untouched: re-reading is
    free, and replaying a stale read would be worse than making the call.

    The schemas the model sees do not change, so this is invisible from the
    prompt's side.
    """
    from dataclasses import replace

    from waku.tools.registry import ToolRegistry

    guarded = ToolRegistry()
    for schema in tools.schemas():
        tool = tools.get(schema["name"])
        if tool is None:
            continue
        if not is_side_effecting(tool.name):
            guarded.register(tool)
            continue

        def wrapped(_tool=tool, **kwargs) -> str:
            # Underscore keys are the loop's own channels (_notify), not part
            # of what the model asked for, so they stay out of the key.
            args = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            return table.guard(thread_id, step_id, _tool.name, args,
                               lambda: _tool.fn(**kwargs))

        guarded.register(replace(tool, fn=wrapped))
    return guarded
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest lab/planning-agent/tests/test_bridge.py lab/planning-agent/tests/test_idempotency.py -q`
Expected: PASS, 16 tests.

- [ ] **Step 8: Prove the read-only pass-through can fail**

Temporarily change `READ_ONLY` to an empty `frozenset()` and re-run.
Expected: `test_a_read_only_tool_is_not_deduped` FAILS, because the second
`list_events` call replays the first answer. Put `READ_ONLY` back, confirm
PASS, and name what you broke in the commit message.

- [ ] **Step 9: Commit**

```bash
git add lab/planning-agent/bridge.py lab/planning-agent/idempotency.py lab/planning-agent/tests/test_bridge.py
git commit -m "$(cat <<'MSG'
Run each planned step as one Waku loop turn

run_loop executes tools itself, so the dedup guard sits on the registry the
loop is handed rather than on the bridge: a side-effecting tool that already
ran under this thread and step returns its recorded result. Read-only tools
pass through, because replaying a stale read is worse than making the call.
MSG
)"
```

---

### Task 8: `session.py` and the resume checks

**Files:**
- Create: `lab/planning-agent/session.py`
- Test: `lab/planning-agent/tests/test_resume.py`

**Interfaces:**
- Consumes: `DedupTable` from idempotency; `build_graph` from graph; `new_state` from state.
- Produces: `open_saver(path) -> tuple[SqliteSaver, sqlite3.Connection]`, `open_dedup(state_db) -> tuple[DedupTable, sqlite3.Connection]`, `thread_config(thread_id, recursion_limit=200) -> dict`.

This is the headline capability and the reason the topic exists. It comes after
Task 5 because running it against side-effecting tools without the dedup table
is how a demonstration double-books a calendar.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_resume.py`:

```python
import sqlite3

import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.graph import build_graph
from lab.planning_agent.idempotency import DedupTable
from lab.planning_agent.session import open_dedup, open_saver, thread_config
from lab.planning_agent.state import new_state


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def _app(saver, log, fail_on=None, world=None, table=None):
    """A graph whose step optionally commits a side effect and then crashes."""
    from lab.planning_agent.state import next_step

    def plan_fn(state):
        return {"plan": _steps("a", "b", "c")}

    def execute_fn(state):
        step = next_step(state)

        def act():
            if world is not None:
                world.append(step["id"])
            return step["id"]

        if table is None:
            act()
        else:
            table.guard("t1", step["id"], "create_event", {"n": step["id"]}, act)
        log.append(step["id"])
        if fail_on and step["id"] == fail_on[0] and fail_on[1]["armed"]:
            raise RuntimeError(f"crash in {step['id']}")
        return {"done": [{"step_id": step["id"], "digest": "", "artifacts": {},
                          "output_path": ""}],
                "budget_spent": state["budget_spent"] + 0.01}

    return build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                       replan_fn=lambda s: {}, synthesize_fn=lambda s: {},
                       checkpointer=saver)


def test_a_finished_task_needs_no_resume(tmp_path):
    saver, conn = open_saver(tmp_path / "ckpt.sqlite")
    log: list[str] = []
    out = _app(saver, log).invoke(new_state("g", 10.0), thread_config("t1"))
    assert log == ["a", "b", "c"]
    assert [r["step_id"] for r in out["done"]] == ["a", "b", "c"]
    conn.close()


def test_a_crashed_task_resumes_in_a_fresh_process_without_redoing_work(tmp_path):
    """The claim the whole topic rests on. The second half uses a brand-new
    connection, which is what a restarted process has."""
    db = tmp_path / "ckpt.sqlite"
    armed = {"armed": True}

    saver1, conn1 = open_saver(db)
    log1: list[str] = []
    with pytest.raises(RuntimeError):
        _app(saver1, log1, fail_on=("b", armed)).invoke(
            new_state("g", 10.0), thread_config("t1"))
    assert log1 == ["a", "b"]
    conn1.close()

    armed["armed"] = False               # the transient cause is gone
    saver2, conn2 = open_saver(db)       # a fresh process
    log2: list[str] = []
    app = _app(saver2, log2, fail_on=("b", armed))
    assert [r["step_id"] for r in app.get_state(thread_config("t1")).values["done"]] == ["a"]
    out = app.invoke(None, thread_config("t1"))
    assert log2 == ["b", "c"], "'a' must not run again"
    assert [r["step_id"] for r in out["done"]] == ["a", "b", "c"]
    conn2.close()


def test_resume_repeats_a_side_effect_unless_the_dedup_table_guards_it(tmp_path):
    """Review of the whole design in one test. The checkpoint stays consistent
    either way; only the dedup table keeps the world consistent."""
    for use_table, expected in ((False, ["a", "b", "b", "c"]), (True, ["a", "b", "c"])):
        db = tmp_path / f"ckpt-{use_table}.sqlite"
        armed = {"armed": True}
        world: list[str] = []
        table = None
        if use_table:
            table, dedup_conn = open_dedup(tmp_path / f"state-{use_table}.db")

        saver1, conn1 = open_saver(db)
        with pytest.raises(RuntimeError):
            _app(saver1, [], fail_on=("b", armed), world=world, table=table).invoke(
                new_state("g", 10.0), thread_config("t1"))
        conn1.close()

        armed["armed"] = False
        saver2, conn2 = open_saver(db)
        _app(saver2, [], fail_on=("b", armed), world=world, table=table).invoke(
            None, thread_config("t1"))
        conn2.close()
        if use_table:
            dedup_conn.close()

        assert world == expected, f"use_table={use_table}"


def test_the_checkpoint_does_not_grow_with_the_number_of_finished_steps(tmp_path):
    """Why full outputs go to disk. A 4KB output per step carried in state
    would be re-serialised on every later step."""
    db = tmp_path / "ckpt.sqlite"
    saver, conn = open_saver(db)
    from lab.planning_agent.state import next_step

    def plan_fn(state):
        return {"plan": _steps(*[f"s{i}" for i in range(12)])}

    def execute_fn(state):
        step = next_step(state)
        return {"done": [{"step_id": step["id"], "digest": "d" * 200,
                          "artifacts": {}, "output_path": "/tmp/x"}],
                "budget_spent": state["budget_spent"] + 0.01}

    app = build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                      replan_fn=lambda s: {}, synthesize_fn=lambda s: {},
                      checkpointer=saver)
    app.invoke(new_state("g", 10.0), thread_config("t1"))
    sizes = [len(str(snap.values)) for snap in app.get_state_history(thread_config("t1"))]
    conn.close()
    growth = max(sizes) - min(sizes)
    assert growth < 12 * 400, (
        "state grew faster than one 200-char digest per step; something large "
        f"is riding in state (growth={growth})")


def test_waku_create_event_reports_already_exists_on_a_replay(tmp_path):
    """Why the planner keys its own calls rather than trusting the tool.

    calendar.py guards by natural key, so a replay is told the event exists.
    That is correct for the local table, and it is also why a replay cannot
    tell 'nothing to do' from 'the downstream syncs never ran'.
    """
    from waku import db
    from waku.tools import calendar

    conn = db.connect(tmp_path)
    tool = calendar.make_tool(conn, tmp_path)
    first = tool.fn(title="lunch", start="2026-10-12T12:00")
    second = tool.fn(title="lunch", start="2026-10-12T12:00")
    conn.close()

    assert "already exists" not in first
    assert "already exists" in second
    assert "not duplicated" in second
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_resume.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.session'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/session.py`:

```python
"""Opening the two databases, and the config a thread runs under.

Two files, two jobs. The checkpointer's own SQLite file holds the task's
progress and may be deleted with the task. The dedup table lives in Waku's
state.db, because it must outlive any single task: it is the record of what
was already done to the world.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from lab.planning_agent.idempotency import DedupTable

# The resolved default in langgraph 1.2.14 is 10007, which is not a guard at
# all. A planned task that needs more than a couple of hundred supersteps has
# gone wrong, and stopping is the right answer.
RECURSION_LIMIT = 200


def open_saver(path: str | Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    """A checkpointer and the connection it owns.

    SqliteSaver.from_conn_string is a context manager, which suits a short
    block and not a long-lived run, so the connection is made here and handed
    back for the caller to close.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver, conn


def open_dedup(state_db: str | Path) -> tuple[DedupTable, sqlite3.Connection]:
    """The dedup table in state.db. Creates its own table and nothing else."""
    conn = sqlite3.connect(Path(state_db), check_same_thread=False)
    table = DedupTable(conn)
    table.setup()
    return table, conn


def thread_config(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_resume.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Add the one test that really kills a process**

The tests above open a second connection in the same interpreter, which is
close to a restart and cheap enough to run every time. The README's headline
claim is stronger than that, so one test earns it properly. Append to
`lab/planning-agent/tests/test_resume.py`:

```python
import os
import subprocess
import sys
import textwrap

WORKER = '''
import os, sqlite3, sys
sys.path.insert(0, {root!r})
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_pa", os.path.join({root!r}, "lab", "planning-agent", "_path.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

from lab.planning_agent.graph import build_graph
from lab.planning_agent.idempotency import DedupTable
from lab.planning_agent.session import open_saver, thread_config
from lab.planning_agent.state import new_state, next_step

ckpt, state_db, world, die = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "die"
saver, conn = open_saver(ckpt)
dedup_conn = sqlite3.connect(state_db)
table = DedupTable(dedup_conn); table.setup()

def plan_fn(state):
    return {{"plan": [{{"id": i, "description": i, "status": "pending"}}
                     for i in ("a", "b", "c")]}}

def execute_fn(state):
    step = next_step(state)
    def act():
        with open(world, "a", encoding="utf-8") as fh:
            fh.write(step["id"] + "\\n")
        return step["id"]
    table.guard("t1", step["id"], "create_event", {{"n": step["id"]}}, act)
    if die and step["id"] == "b":
        # No exception, no unwinding, no checkpoint: the process is simply gone,
        # which is what a kill -9 or a closed laptop lid does.
        os._exit(1)
    return {{"done": [{{"step_id": step["id"], "digest": "", "artifacts": {{}},
                      "output_path": ""}}],
            "budget_spent": state["budget_spent"] + 0.01}}

app = build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                  replan_fn=lambda s: {{}}, synthesize_fn=lambda s: {{}},
                  checkpointer=saver)
app.invoke(new_state("g", 10.0) if sys.argv[5] == "start" else None,
           thread_config("t1"))
conn.close(); dedup_conn.close()
'''


def test_a_killed_process_resumes_without_repeating_the_side_effect(tmp_path):
    """The claim the README makes, tested the way the README words it. The
    first process does not raise: it calls os._exit(1) after the side effect has
    committed and before the checkpoint does, so nothing unwinds."""
    root = str(Path(__file__).resolve().parents[3])
    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent(WORKER).format(root=root), encoding="utf-8")
    ckpt, state_db, world = (str(tmp_path / "ckpt.sqlite"),
                             str(tmp_path / "state.db"), str(tmp_path / "world.txt"))

    first = subprocess.run([sys.executable, str(worker), ckpt, state_db, world,
                            "die", "start"], capture_output=True, text=True)
    assert first.returncode == 1, f"the worker should have died: {first.stderr}"
    assert Path(world).read_text().split() == ["a", "b"]

    second = subprocess.run([sys.executable, str(worker), ckpt, state_db, world,
                             "live", "resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert Path(world).read_text().split() == ["a", "b", "c"], (
        "'b' committed before the process died and must not commit again")
```

Add `from pathlib import Path` to the module's imports if it is not there yet.

Run: `python -m pytest lab/planning-agent/tests/test_resume.py -q`
Expected: PASS, 6 tests. This one spawns two interpreters, so it is the slowest
in the topic; it is also the only one that proves the claim without a caveat.

- [ ] **Step 6: Commit**

```bash
git add lab/planning-agent/session.py lab/planning-agent/tests/test_resume.py
git commit -m "$(cat <<'MSG'
Resume a crashed task in a fresh process

A crashed run picks up from the last completed step with a new connection,
which is what a restarted process has. The paired test shows the dedup table
is what keeps the world consistent: without it the same run leaves
['a','b','b','c'] behind, with it ['a','b','c'].

One test goes further and really kills an interpreter with os._exit(1) after a
side effect has committed and before the checkpoint has, then resumes in a
second one. That is the claim the README makes, so it is worth the two process
spawns it costs.
MSG
)"
```

---

### Task 9: the replanner, and lessons that get read

**Files:**
- Create: `lab/planning-agent/replan.py`
- Test: `lab/planning-agent/tests/test_replan.py`

**Interfaces:**
- Consumes: `PlanState`, `Revision`, `Step`, `next_step` from state.
- Produces: `make_replanner(propose) -> NodeFn` where `propose(goal, remaining, done, lessons) -> list[Step]`, and `preserved(state) -> list[StepResult]`.

The replanner restructures only what has not happened. `done` is append-only,
so it cannot do otherwise, which is the reason for that reducer.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_replan.py`:

```python
import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.replan import make_replanner
from lab.planning_agent.state import new_state


def _steps(*ids):
    return [{"id": i, "description": i, "status": "pending"} for i in ids]


def _result(step_id):
    return {"step_id": step_id, "digest": step_id, "artifacts": {}, "output_path": ""}


def _state_with_a_failure():
    state = new_state("g", budget_limit=10.0)
    state["plan"] = _steps("a", "b", "c")
    state["done"] = [_result("a")]
    state["lessons"] = [{"step_id": "b", "attempt": "called the venue",
                         "failure": "line was dead", "diagnosis": "use email"}]
    state["replan_requested"] = True
    state["last_error"] = "RuntimeError: line was dead"
    return state


def test_replanning_keeps_finished_work_and_restructures_the_rest():
    merged = make_replanner(lambda **kw: _steps("b2", "c"))(_state_with_a_failure())
    assert [s["id"] for s in merged["plan"]] == ["b2", "c"]
    assert "done" not in merged, "finished work is never rewritten"


def test_replanning_clears_the_request_so_the_cycle_can_continue():
    merged = make_replanner(lambda **kw: _steps("b2"))(_state_with_a_failure())
    assert merged["replan_requested"] is False


def test_replanning_appends_a_revision_saying_why():
    merged = make_replanner(lambda **kw: _steps("b2"))(_state_with_a_failure())
    revision = merged["revisions"][0]
    assert revision["after_step"] == "a"
    assert "b2" in revision["change"]
    assert "line was dead" in revision["reason"]


def test_the_replanner_is_given_the_lessons():
    """Review of gap 3: a Lesson nothing reads is dead weight."""
    seen = {}

    def propose(*, goal, remaining, done, lessons):
        seen.update(lessons=lessons, remaining=[s["id"] for s in remaining],
                    done=[r["step_id"] for r in done], goal=goal)
        return _steps("b2")

    make_replanner(propose)(_state_with_a_failure())
    assert seen["goal"] == "g"
    assert seen["remaining"] == ["b", "c"], "only unfinished steps are up for revision"
    assert seen["done"] == ["a"]
    assert seen["lessons"][0]["failure"] == "line was dead"


def test_a_replan_that_proposes_nothing_leaves_no_work_and_ends_the_task():
    merged = make_replanner(lambda **kw: [])(_state_with_a_failure())
    assert merged["plan"] == []


def test_a_replan_may_legitimately_shorten_the_plan_below_what_is_done():
    """The case Review Focus 1 guards. Two steps are finished, the replanner
    returns one step, and next_step must still answer sensibly."""
    from lab.planning_agent.state import next_step

    state = _state_with_a_failure()
    state["done"] = [_result("a"), _result("b")]
    merged = make_replanner(lambda **kw: _steps("c"))(state)
    state["plan"] = merged["plan"]
    assert next_step(state)["id"] == "c"


def test_every_replan_counts_towards_the_cap():
    """Review Focus 7. The count is what a renamed step cannot evade."""
    merged = make_replanner(lambda **kw: _steps("b2"))(_state_with_a_failure())
    assert merged["replans"] == 1


def test_a_replan_that_raises_still_counts():
    def boom(**kw):
        raise RuntimeError("no")

    assert make_replanner(boom)(_state_with_a_failure())["replans"] == 1


def test_a_failing_proposer_leaves_the_plan_alone_and_records_the_failure():
    def boom(**kw):
        raise RuntimeError("the planner model refused")

    merged = make_replanner(boom)(_state_with_a_failure())
    assert "plan" not in merged, "a broken replanner must not wipe the plan"
    assert merged["replan_requested"] is False, "and must not loop forever"
    assert "refused" in merged["last_error"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_replan.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.planning_agent.replan'`.

- [ ] **Step 3: Write the minimal implementation**

Write `lab/planning-agent/replan.py`:

```python
"""Revising the plan without losing what already happened.

Roll back the work, keep the reflection. A failed step's output is discarded;
the lesson it produced is not, and the replanner is handed those lessons along
with the unfinished steps. Without that it is invoked at exactly the moment
nothing records what went wrong, and it can propose the failing approach again.

`done` is append-only, so this function cannot rewrite history even by mistake.
It returns a new `plan` and one `Revision`, and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable

from lab.planning_agent.state import PlanState, Revision, Step, StepResult

Propose = Callable[..., list[Step]]


def preserved(state: PlanState) -> list[StepResult]:
    """What the replanner may not touch."""
    return list(state["done"])


def _remaining(state: PlanState) -> list[Step]:
    finished = {result["step_id"] for result in state["done"]}
    return [step for step in state["plan"] if step["id"] not in finished]


def make_replanner(propose: Propose) -> Callable[[PlanState], dict]:
    """A replan node. `propose` returns the steps that should replace the
    unfinished ones; it never sees a chance to change the finished ones."""

    def replan(state: PlanState) -> dict:
        done = preserved(state)
        last = done[-1]["step_id"] if done else ""
        try:
            steps = propose(goal=state["goal"], remaining=_remaining(state),
                            done=done, lessons=list(state["lessons"]))
        except Exception as exc:
            # A broken replanner must not wipe the plan and must not leave the
            # request set, which would route straight back here for ever. The
            # attempt still counts: a replanner that raises every time is
            # exactly what MAX_REPLANS exists to stop.
            return {"replan_requested": False,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "replans": state["replans"] + 1}

        return {"plan": list(steps),
                "replan_requested": False,
                "last_error": "",
                # Counted here, not in the router: this is where a replan
                # actually happened. after_step reads it and stops at
                # MAX_REPLANS, which is the bound a replanner cannot evade by
                # renaming the step that keeps failing.
                "replans": state["replans"] + 1,
                "revisions": [Revision(
                    after_step=last,
                    change="remaining steps are now "
                           + (", ".join(s["id"] for s in steps) or "(none)"),
                    reason=state["last_error"] or "new work was discovered")]}

    return replan
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest lab/planning-agent/tests/test_replan.py -q`
Expected: PASS, 9 tests.

- [ ] **Step 5: Prove the loop guard can fail**

Temporarily remove `"replan_requested": False` from the exception branch and
re-run `test_a_failing_proposer_leaves_the_plan_alone_and_records_the_failure`.
Expected: FAIL. Put it back and say in the commit message what you broke — a
replanner that leaves the flag set routes back to itself until the recursion
limit stops it.

- [ ] **Step 6: Commit**

```bash
git add lab/planning-agent/replan.py lab/planning-agent/tests/test_replan.py
git commit -m "Revise the plan from the failure point, keeping the lessons"
```

---

### Task 10: rollups and the measurements

**Files:**
- Create: `lab/planning-agent/run.py`
- Modify: `lab/planning-agent/graph.py` (add the `roll_up` node)
- Test: `lab/planning-agent/tests/test_rollup_flow.py`

**Interfaces:**
- Consumes: everything built so far.
- Produces: `maybe_roll_up(state, every=8) -> dict` in `graph.py`; `measure(steps, roll_up) -> list[dict]` and a `main()` CLI in `run.py`.

- [ ] **Step 1: Write the failing test**

Write `lab/planning-agent/tests/test_rollup_flow.py`:

```python
import pytest

pytest.importorskip("langgraph")

from lab.planning_agent.graph import maybe_roll_up
from lab.planning_agent.run import measure
from lab.planning_agent.state import new_state


def _result(step_id, digest="d" * 120):
    return {"step_id": step_id, "digest": digest, "artifacts": {}, "output_path": ""}


def test_nothing_is_rolled_up_before_the_threshold():
    state = new_state("g", 10.0)
    state["done"] = [_result(f"s{i}") for i in range(3)]
    assert maybe_roll_up(state, every=8) == {}


def test_reaching_the_threshold_collapses_the_finished_steps_into_one_rollup():
    state = new_state("g", 10.0)
    state["done"] = [_result(f"s{i}") for i in range(8)]
    merged = maybe_roll_up(state, every=8)
    rollup = merged["rollups"][0]
    assert rollup["step_ids"] == [f"s{i}" for i in range(8)]
    assert "s0" in rollup["summary"]


def test_steps_already_inside_a_rollup_are_not_rolled_up_again():
    state = new_state("g", 10.0)
    state["done"] = [_result(f"s{i}") for i in range(10)]
    state["rollups"] = [{"step_ids": [f"s{i}" for i in range(8)], "summary": "earlier"}]
    assert maybe_roll_up(state, every=8) == {}, "only 2 steps are unrolled"

    state["done"].extend(_result(f"s{i}") for i in range(10, 16))
    merged = maybe_roll_up(state, every=8)
    assert merged["rollups"][0]["step_ids"] == [f"s{i}" for i in range(8, 16)]


def test_rollups_slow_prompt_growth_without_flattening_it():
    """Measured, not claimed. Rollups cut the final prompt by about 2.2x over
    24 steps; they do not flatten growth, because one rollup accumulates per
    eight steps. The ratio is asserted with headroom so an unrelated prompt
    tweak does not redden this, and `measure` reads its digest size from
    digest.py rather than hardcoding one: a measurement that drifts from the
    real prompt path measures nothing."""
    without = measure(steps=24, roll_up=False)
    with_rollups = measure(steps=24, roll_up=True)

    assert without[-1]["prompt_chars"] > without[0]["prompt_chars"] * 3, (
        "without rollups the prompt should grow substantially")
    ratio = without[-1]["prompt_chars"] / with_rollups[-1]["prompt_chars"]
    assert ratio > 1.8, f"rollups should cut the final prompt well past a fifth: {ratio:.2f}x"

    # Still linear, just cheaper. Doubling the steps roughly doubles the
    # prompt even with rollups on, because rollups accumulate at one per eight
    # steps. Asserting anything better than this would be asserting a property
    # the design does not have: the first version of this test claimed
    # sublinear growth and measured 3,577 -> 7,422, which is 2.07x.
    short = measure(steps=12, roll_up=True)[-1]["prompt_chars"]
    long = with_rollups[-1]["prompt_chars"]
    assert 1.8 < long / short < 2.4, (
        f"growth with rollups is linear in steps, not flat: {short} -> {long}")


def test_the_measurement_uses_the_same_digest_size_as_the_bridge():
    """The reason the old assertion failed. Digesting at 120 chars while a
    rollup caps at 800 makes eight digests and one rollup almost the same size,
    so the mechanism looked worthless when it was the measurement that was
    wrong."""
    from lab.planning_agent import digest as digest_module

    rows = measure(steps=2, roll_up=False)
    assert digest_module.STEP_DIGEST_CHARS == 600, (
        "if this changes, the ratio above moves and the README numbers are stale")
    assert rows[1]["prompt_chars"] > rows[0]["prompt_chars"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest lab/planning-agent/tests/test_rollup_flow.py -q`
Expected: FAIL — `ImportError: cannot import name 'maybe_roll_up'`.

- [ ] **Step 3: Add `maybe_roll_up` to `graph.py`**

Append to `lab/planning-agent/graph.py`:

```python
ROLL_UP_EVERY = 8


def _unrolled(state: PlanState) -> list:
    inside = {sid for rollup in state["rollups"] for sid in rollup["step_ids"]}
    return [r for r in state["done"] if r["step_id"] not in inside]


def maybe_roll_up(state: PlanState, every: int = ROLL_UP_EVERY) -> dict:
    """Collapse a finished phase once enough steps have piled up.

    A one-line digest per step still grows linearly, so past a couple of dozen
    steps the digest list alone crowds the prompt. The outputs stay on disk, so
    this shortens the prompt and loses nothing.
    """
    from lab.planning_agent.digest import make_rollup

    pending = _unrolled(state)
    if len(pending) < every:
        return {}
    return {"rollups": [make_rollup(pending[:every])]}
```

Then wire it in: inside `build_graph`, replace the `execute_step` conditional
edges block with a `roll_up` node between the step and the router, so every
step passes through it:

```python
    graph.add_node("roll_up", lambda state: maybe_roll_up(state))
    graph.add_edge("execute_step", "roll_up")
    graph.add_conditional_edges(
        "roll_up", after_step,
        {"step": "execute_step", "replan": "replan", "synthesize": "synthesize"})
```

Delete the old `graph.add_conditional_edges("execute_step", after_step, ...)`
call. `after_step` is unchanged: it still reads the same state, only from one
node later.

- [ ] **Step 4: Write `run.py`**

Write `lab/planning-agent/run.py`:

```python
"""The topic's own checks. Not part of make gate: lab never is.

`measure` runs the graph with stub nodes and records how big each step's prompt
would be. No model is called, so the numbers are about structure rather than
about a provider on the day.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

# The hyphen in the directory name means `lab.planning_agent` does not exist
# until the alias is installed, and no conftest runs for a plain `python`
# invocation. This import has to come before any lab.planning_agent one, which
# is why it is not sorted with them.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "lab_planning_agent_path", Path(__file__).resolve().parent / "_path.py")
_path_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_path_module)

from lab.planning_agent.bridge import step_prompt  # noqa: E402
from lab.planning_agent.graph import build_graph, maybe_roll_up
from lab.planning_agent.session import open_saver, thread_config
from lab.planning_agent.state import new_state, next_step

STEP_OUTPUT = "x" * 3000
# measure() appends to `done` and never removes from it, because the reducer
# does not allow removal. The prompt gets shorter only because _history skips
# steps that a rollup already covers.


def measure(steps: int, roll_up: bool) -> list[dict]:
    """Prompt size per step, with and without rollups.

    Two rules keep this honest. The digest size comes from digest.py, the same
    one the bridge uses, because a measurement at a different size measures a
    system nobody runs. And `done` is only ever appended to, exactly as the
    reducer requires: an earlier version of this function deleted rolled-up
    entries from it, which no graph can do, so it measured a prompt the real
    one never produced. The shortening comes from `_history` filtering, which
    is what step_prompt actually does.
    """
    from lab.planning_agent.digest import STEP_DIGEST_CHARS, step_digest

    rows: list[dict] = []
    state = new_state("measure prompt growth", budget_limit=1e9)
    state["plan"] = [{"id": f"s{i}", "description": f"step {i}", "status": "pending"}
                     for i in range(steps)]
    for _ in range(steps):
        step = next_step(state)
        if step is None:
            break
        rows.append({"step": step["id"], "prompt_chars": len(step_prompt(state, step))})
        state["done"] = state["done"] + [{
            "step_id": step["id"],
            "digest": step_digest(STEP_OUTPUT, limit=STEP_DIGEST_CHARS),
            "artifacts": {}, "output_path": ""}]
        if roll_up:
            merged = maybe_roll_up(state)
            if merged:
                state["rollups"] = state["rollups"] + merged["rollups"]
    return rows


def _report(steps: int) -> None:
    without = measure(steps, roll_up=False)
    with_rollups = measure(steps, roll_up=True)
    print(f"{'step':>6}  {'no rollups':>12}  {'rollups':>12}")
    for a, b in zip(without, with_rollups):
        print(f"{a['step']:>6}  {a['prompt_chars']:>12,}  {b['prompt_chars']:>12,}")
    print(f"\nfinal prompt: {without[-1]['prompt_chars']:,} chars without rollups, "
          f"{with_rollups[-1]['prompt_chars']:,} with")


def _smoke(steps: int) -> None:
    """One end-to-end run with stub nodes, through a real SQLite checkpointer."""
    with tempfile.TemporaryDirectory() as tmp:
        saver, conn = open_saver(Path(tmp) / "ckpt.sqlite")

        def plan_fn(state):
            return {"plan": [{"id": f"s{i}", "description": f"step {i}",
                              "status": "pending"} for i in range(steps)]}

        def execute_fn(state):
            step = next_step(state)
            return {"done": [{"step_id": step["id"], "digest": "ok",
                              "artifacts": {}, "output_path": ""}],
                    "budget_spent": state["budget_spent"] + 0.001}

        app = build_graph(plan_fn=plan_fn, execute_fn=execute_fn,
                          replan_fn=lambda s: {}, synthesize_fn=lambda s: {},
                          checkpointer=saver)
        out = app.invoke(new_state("smoke", 10.0), thread_config("smoke"))
        conn.close()
        print(f"ran {len(out['done'])} steps, {len(out['rollups'])} rollups, "
              f"spent {out['budget_spent']:.4f} USD")


def main() -> None:
    parser = argparse.ArgumentParser(description="planning-agent lab checks")
    parser.add_argument("--measure", type=int, metavar="STEPS",
                        help="report prompt growth over STEPS steps")
    parser.add_argument("--smoke", type=int, metavar="STEPS",
                        help="run the graph end to end with stub nodes")
    args = parser.parse_args()
    if args.measure:
        _report(args.measure)
    elif args.smoke:
        _smoke(args.smoke)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest lab/planning-agent/tests -q`
Expected: PASS, all modules. The rollup wiring changed `graph.py`, so
`test_graph.py` must still pass too — if `test_the_cycle_runs_every_step_once_and_stops`
fails, the `roll_up` node is not on the path between `execute_step` and the
router.

- [ ] **Step 6: Run the CLI and keep the numbers**

Run: `python lab/planning-agent/run.py --measure 24`
Run: `python lab/planning-agent/run.py --smoke 20`
Expected: a table where the no-rollups column grows step by step and the
rollups column grows far more slowly, stepping up once per rollup rather than
resetting. Over 24 steps expect roughly 16,000 chars without rollups against
roughly 7,400 with them, a ratio near 2.2x. Copy the real final line into Task
11; if the ratio is below 1.8 the test above has already failed and the cause
is a digest or rollup size that no longer match.

- [ ] **Step 7: Commit**

```bash
git add lab/planning-agent/run.py lab/planning-agent/graph.py lab/planning-agent/tests/test_rollup_flow.py
git commit -m "Collapse finished phases to cut the prompt by about 2.2x"
```

---

### Task 11: the findings and the graduation verdict

**Files:**
- Modify: `lab/planning-agent/README.md`
- Modify: `docs/status.md`

**Interfaces:**
- Consumes: the measurements from Task 10 Step 6.
- Produces: nothing code depends on.

- [ ] **Step 1: Rewrite "What we found" with the real numbers**

Replace the placeholder section in `lab/planning-agent/README.md`. Use the
numbers you actually recorded; the shape is:

```markdown
## What we found

Measured on langgraph 1.2.14 with stub nodes, so these are properties of the
structure and not of a provider on the day.

- **Resume works across a process boundary.** A run killed in step 2 of 3
  picked up at step 2 with a brand-new SQLite connection and did not repeat
  step 1.
- **Resume re-runs the failed node, so a side effect commits twice.** The same
  three-step run left `['a','b','b','c']` in an external list while its
  checkpointed state read `['a','b','c']`. A dedup table keyed by thread, step,
  tool and arguments brings the world back to `['a','b','c']`.
- **Waku's own guards are uneven.** `create_event` dedupes by title and start;
  `send_message` and `save_note` do not. And `create_event`'s guard runs before
  the ics write and the Apple and Google syncs, so a replay is told "already
  exists" and those syncs never happen.
- **Rollups slow prompt growth; they do not flatten it.** <N> chars at the
  last of 24 steps without rollups, <M> with, a ratio of <R>x. Growth over the
  run falls from about 18x to about 8x. Flat growth would need a second
  compaction tier, a rollup of rollups, which this topic did not build: there
  is no measurement past 24 steps to justify compounding the summarising loss.
- **A rollup only shortens a prompt if the digests it replaced leave.** `done`
  is append-only, so nothing can be removed from it; the prompt builder has to
  skip the steps a rollup already covers. Built the other way, a rollup makes
  the prompt longer, and a measurement that edits `done` directly will not
  show it.
- **Checkpoint size** did not grow with the number of finished steps once full
  outputs moved to the scratch directory.
- **The budget is not a termination bound.** A step whose loop raises before
  its first model call costs nothing, so `budget_spent` never moves and the
  step is selected again indefinitely. Two bounds fixed it: a step with two
  lessons against it is skipped, and the task caps total replans. The per-step
  bound alone is evadable, because a replanner that renames a failed step
  resets its lesson count.
- **A dedup table has to record failures as failures.** `ToolRegistry.execute`
  returns a tool's exception as an ordinary string, so a table that stores
  results without a status replays a timeout for ever. Recording every outcome
  with `ok` or `error`, and replaying only `ok`, keeps the crash window closed
  while leaving a transient failure retryable.
```

- [ ] **Step 2: Write the graduation verdict**

Replace the `## Graduation` section with a decision, not a maybe. The question
is narrow: does `waku/graph/` need LangGraph, or does it need a checkpointer?
Say which, and at which rung of the footprint ladder the piece would land. If
the answer is "copy the idea, not the dependency", say what the smaller thing
is — a `thread_id`, a state blob and a step cursor in `state.db`.

Record separately that the idempotency finding is independent of LangGraph:
`create_event`'s skip-the-syncs hole is a `waku/` bug on its own terms and is
worth an issue whatever happens to this topic.

- [ ] **Step 3: Add a line to `docs/status.md`**

`status.md` is rewritten whole, never appended to, and capped at 120 lines.
Add one row to "What is deliberately not built" or a line under "Open
questions" naming the planning topic and where its verdict lives. Keep the
`**Last updated:**` line current — `test_status_is_short_and_dated` checks both
the cap and the date.

- [ ] **Step 4: Run the full check set**

```bash
python -m pytest lab/planning-agent/tests -q
python -m pytest evals/deterministic -q
make lint
```

Expected: lab tests pass; the deterministic tier passes; ruff is clean. Note
that `make lint` covers `waku evals scripts hosted` and not `lab`, so lab code
is not linted — keep it tidy by hand.

- [ ] **Step 5: Commit**

```bash
git add lab/planning-agent/README.md docs/status.md
git commit -m "Record the planning topic's findings and graduation verdict"
```
