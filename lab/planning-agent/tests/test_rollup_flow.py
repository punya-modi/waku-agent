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
