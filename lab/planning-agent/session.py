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

# The resolved default in langgraph 1.2 is 10007, which is not a guard at all.
# A planned task that needs more than a couple of hundred supersteps has gone
# wrong, and stopping is the right answer.
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
