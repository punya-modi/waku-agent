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
