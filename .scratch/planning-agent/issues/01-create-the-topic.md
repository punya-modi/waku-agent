# 01 — Create the topic, pin the versions, keep CI green

Status: done
Blocked by: none
Plan task: Task 1 in `docs/superpowers/plans/2026-10-11-planning-agent.md`

The directory and its playbook README are inseparable: `test_lab_topics_follow_the_playbook`
fails the moment `lab/planning-agent/` exists without all six playbook headings
and a `Verified against: <tool> <version>, <date>` line.

## Done when

- `lab/planning-agent/` exists with `README.md`, `__init__.py`, `tests/__init__.py`
- The README carries all six headings: The question, What we connect, Run it, What we found, Video angle, Graduation
- The `Verified against:` line names the versions actually resolved, not the ones in the plan
- `pytest evals/deterministic/test_rulebook.py` passes, 17 tests
- `pytest lab/planning-agent` reports "no tests ran" with no collection error
- A row for the topic is added to `lab/README.md`'s table (convention, not CI)
