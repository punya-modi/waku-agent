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
from lab.planning_agent.graph import build_graph, maybe_roll_up  # noqa: E402
from lab.planning_agent.session import open_saver, thread_config  # noqa: E402
from lab.planning_agent.state import new_state, next_step  # noqa: E402

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
