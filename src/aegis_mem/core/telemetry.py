"""Asynchronous SQLite audit and telemetry logger for memory operations and adaptive utility scoring."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_mem.config import settings


class TelemetryLogger:
    """Non-blocking SQLite telemetry store tracking recall latency, errors, and adaptive rule utility."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.telemetry_db_path
        self._initialized = False
        self._persistent_conn: sqlite3.Connection | None = None
        if self._db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)

    def _get_connection(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        path_obj = Path(self._db_path)
        if path_obj.parent and str(path_obj.parent) != ".":
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _ensure_tables_sync(self) -> None:
        if self._initialized:
            return
        conn = self._get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    latency_ms REAL DEFAULT 0.0,
                    success INTEGER NOT NULL,
                    details_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_utility (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    utility_score REAL DEFAULT 1.0,
                    last_accessed TEXT NOT NULL
                )
                """
            )
            conn.commit()
            self._initialized = True
        finally:
            if self._persistent_conn is None:
                conn.close()

    def _log_event_sync(
        self,
        event_type: str,
        agent_id: str,
        latency_ms: float,
        success: bool,
        details: dict[str, Any] | None,
    ) -> None:
        self._ensure_tables_sync()
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO telemetry_events (timestamp, event_type, agent_id, latency_ms, success, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    event_type,
                    agent_id,
                    round(latency_ms, 3),
                    1 if success else 0,
                    json.dumps(details or {}, default=str),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def _record_utility_sync(
        self, memory_id: str, agent_id: str, succeeded: bool
    ) -> None:
        self._ensure_tables_sync()
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        inc_success = 1 if succeeded else 0
        try:
            conn.execute(
                """
                INSERT INTO memory_utility (
                    memory_id, agent_id, hit_count, success_count, failure_count, utility_score, last_accessed
                )
                VALUES (?, ?, 1, ?, 0, 1.0, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    hit_count = hit_count + 1,
                    success_count = success_count + excluded.success_count,
                    last_accessed = excluded.last_accessed
                """,
                (memory_id, agent_id, inc_success, now),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def _update_rule_outcome_sync(
        self, memory_id: str, agent_id: str, succeeded: bool
    ) -> float:
        """Apply adaptive utility delta (+0.1 on success capped at 2.0; -0.4 on failure) from base 1.0."""
        self._ensure_tables_sync()
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        delta = 0.1 if succeeded else -0.4
        try:
            row = conn.execute(
                "SELECT utility_score, hit_count, success_count, failure_count FROM memory_utility WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                new_score = min(2.0, round(1.0 + delta, 4))
                conn.execute(
                    """
                    INSERT INTO memory_utility (
                        memory_id, agent_id, hit_count, success_count, failure_count, utility_score, last_accessed
                    )
                    VALUES (?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        agent_id,
                        1 if succeeded else 0,
                        0 if succeeded else 1,
                        new_score,
                        now,
                    ),
                )
            else:
                current_score = float(row[0] if row[0] is not None else 1.0)
                new_score = min(2.0, round(current_score + delta, 4))
                conn.execute(
                    """
                    UPDATE memory_utility
                    SET hit_count = hit_count + 1,
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        utility_score = ?,
                        last_accessed = ?
                    WHERE memory_id = ?
                    """,
                    (
                        1 if succeeded else 0,
                        0 if succeeded else 1,
                        new_score,
                        now,
                        memory_id,
                    ),
                )
            conn.commit()
            return new_score
        finally:
            if self._persistent_conn is None:
                conn.close()

    def _get_utility_score_sync(self, memory_id: str) -> float:
        self._ensure_tables_sync()
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT utility_score FROM memory_utility WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else 1.0
        finally:
            if self._persistent_conn is None:
                conn.close()

    async def log_event(
        self,
        event_type: str,
        agent_id: str,
        latency_ms: float = 0.0,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Asynchronously record a telemetry event without blocking the event loop."""
        await asyncio.to_thread(
            self._log_event_sync, event_type, agent_id, latency_ms, success, details
        )

    async def record_utility(
        self, memory_id: str, agent_id: str, succeeded: bool = True
    ) -> None:
        """Asynchronously update the recall hit count of a procedural rule."""
        await asyncio.to_thread(
            self._record_utility_sync, memory_id, agent_id, succeeded
        )

    async def update_rule_outcome(
        self, memory_id: str, agent_id: str, succeeded: bool
    ) -> float:
        """Asynchronously update a rule's utility score (+0.1 on success max 2.0, -0.4 on failure) and return it."""
        return await asyncio.to_thread(
            self._update_rule_outcome_sync, memory_id, agent_id, succeeded
        )

    async def get_utility_score(self, memory_id: str) -> float:
        """Asynchronously retrieve the current utility score for `memory_id` (defaults to 1.0)."""
        return await asyncio.to_thread(self._get_utility_score_sync, memory_id)


telemetry = TelemetryLogger()
