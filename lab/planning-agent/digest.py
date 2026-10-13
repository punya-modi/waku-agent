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
