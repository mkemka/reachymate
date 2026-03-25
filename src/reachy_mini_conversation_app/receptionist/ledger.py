"""Local SQLite ledger: single source of truth for member point balances.

Non-negotiable guarantees
--------------------------
* No cloud calls — pure local SQLite.
* Idempotent: the same ``interaction_id`` always returns the same result.
* Atomic: deductions happen inside a single SQLite transaction; no partial state.
* No failure path ever deducts points.
"""

from __future__ import annotations

import time
import logging
import threading
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Ledger:
    """Thread-safe local SQLite ledger for member point balances.

    Schema
    ------
    members(member_id PK, display_name, created_at)
    balances(member_id PK FK, points, last_updated)
    transactions(interaction_id PK, member_id FK, cost_points, timestamp, ok, reason)
    """

    def __init__(self, db_path: str | Path, initial_balance: int = 100) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_balance = initial_balance
        self._lock = threading.Lock()
        self._init_db()
        logger.info("Ledger initialised at %s (initial_balance=%d)", self.db_path, initial_balance)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS members (
                    member_id   TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS balances (
                    member_id   TEXT PRIMARY KEY,
                    points      INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    interaction_id TEXT PRIMARY KEY,
                    member_id      TEXT NOT NULL,
                    cost_points    INTEGER NOT NULL,
                    timestamp      REAL NOT NULL,
                    ok             INTEGER NOT NULL,  -- 1=success 0=fail
                    reason         TEXT NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tx_member ON transactions(member_id);
            """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_member(self, member_id: str, display_name: str) -> None:
        """Create member + balance row if they do not exist yet (idempotent)."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO members(member_id, display_name, created_at) VALUES(?,?,?)",
                (member_id, display_name, time.time()),
            )
            conn.execute(
                "INSERT OR IGNORE INTO balances(member_id, points, last_updated) VALUES(?,?,?)",
                (member_id, self.initial_balance, time.time()),
            )
        logger.debug("ensure_member: %s (%s)", member_id, display_name)

    def get_balance(self, member_id: str) -> int | None:
        """Return current points balance, or ``None`` if member not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT points FROM balances WHERE member_id=?", (member_id,)
            ).fetchone()
        return int(row["points"]) if row else None

    def deduct(
        self,
        member_id: str,
        interaction_id: str,
        cost_points: int,
    ) -> dict[str, Any]:
        """Atomic idempotent point deduction.

        Returns ``{"ok": bool, "reason": "none" | "insufficient_points" | "duplicate" | "error"}``.

        Idempotency: if the same ``interaction_id`` is submitted again, the stored
        result is returned immediately without touching the balance.

        No partial updates: either the full deduction commits or nothing changes.
        """
        with self._lock, self._connect() as conn:
            # --- Idempotency check ---
            existing = conn.execute(
                "SELECT ok, reason FROM transactions WHERE interaction_id=?",
                (interaction_id,),
            ).fetchone()
            if existing is not None:
                result = {"ok": bool(existing["ok"]), "reason": existing["reason"]}
                logger.debug("Ledger idempotent hit for %s → %s", interaction_id, result)
                return result

            # --- Balance check ---
            row = conn.execute(
                "SELECT points FROM balances WHERE member_id=?", (member_id,)
            ).fetchone()

            if row is None:
                result = {"ok": False, "reason": "error"}
                logger.warning("Ledger deduct: member %s not found", member_id)
            elif int(row["points"]) < cost_points:
                result = {"ok": False, "reason": "insufficient_points"}
                logger.info(
                    "Ledger deduct: insufficient points (member=%s, have=%d, need=%d)",
                    member_id, int(row["points"]), cost_points,
                )
            else:
                # --- Atomic deduction ---
                conn.execute(
                    "UPDATE balances SET points = points - ?, last_updated = ? WHERE member_id = ?",
                    (cost_points, time.time(), member_id),
                )
                result = {"ok": True, "reason": "none"}
                logger.info(
                    "Ledger deduct: %d pts from %s (interaction=%s)",
                    cost_points, member_id, interaction_id,
                )

            # --- Record transaction (always, even on failure) ---
            conn.execute(
                """INSERT INTO transactions
                   (interaction_id, member_id, cost_points, timestamp, ok, reason)
                   VALUES (?,?,?,?,?,?)""",
                (interaction_id, member_id, cost_points, time.time(), int(result["ok"]), result["reason"]),
            )

        return result

    def add_points(self, member_id: str, points: int) -> bool:
        """Add points to a member's balance (e.g. admin top-up). Returns True on success."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE balances SET points = points + ?, last_updated = ? WHERE member_id = ?",
                (points, time.time(), member_id),
            )
        if cur.rowcount == 0:
            logger.warning("add_points: member %s not found", member_id)
            return False
        logger.info("add_points: +%d to %s", points, member_id)
        return True

    def list_members(self) -> list[dict[str, Any]]:
        """Return all members with their current balance."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT m.member_id, m.display_name, m.created_at, COALESCE(b.points, 0) AS points
                   FROM members m
                   LEFT JOIN balances b ON m.member_id = b.member_id
                   ORDER BY m.display_name""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_member(self, member_id: str) -> dict[str, Any] | None:
        """Return a single member with balance, or None."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT m.member_id, m.display_name, m.created_at, COALESCE(b.points, 0) AS points
                   FROM members m
                   LEFT JOIN balances b ON m.member_id = b.member_id
                   WHERE m.member_id = ?""",
                (member_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_transactions(self, member_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent transactions for a member (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT interaction_id, cost_points, timestamp, ok, reason
                   FROM transactions WHERE member_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (member_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
