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
