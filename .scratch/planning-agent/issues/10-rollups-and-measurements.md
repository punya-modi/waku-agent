# 10 — rollups and the measurements

Status: ready-for-agent
Blocked by: 03, 06, 07
Plan task: Task 10 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

Also adds the `roll_up` node to the graph, between the step and the router.

Do not let `measure` edit `done`. An earlier version did, which no
reducer-backed graph can, so it measured a prompt the real one never produced.
The shortening has to come from ticket 07's `_history` filter.

## Done when

- Nothing rolls up before the threshold; reaching it collapses the finished steps
- Steps already inside a rollup are not rolled up again
- `measure` appends to `done` and never removes, and reads its digest size from `digest.py`
- The ratio without rollups over with is above 1.8 (expect about 2.2x at 24 steps)
- Growth with rollups is asserted LINEAR, between 1.8x and 2.4x from 12 steps to 24. It is not flat, and a test claiming otherwise measured 3,577 to 7,422 and failed
- `run.py` loads `_path.py` by file path, so `python lab/planning-agent/run.py` works on a clean shell
- `--measure 24` and `--smoke 20` both run; keep the final numbers for ticket 11
- All lab tests pass, `test_graph.py` included, with the `roll_up` node on the path
