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
