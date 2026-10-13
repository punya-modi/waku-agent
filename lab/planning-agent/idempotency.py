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
    skipped - a booking that silently never happens.
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
