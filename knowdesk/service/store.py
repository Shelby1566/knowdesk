"""Thread-safe SQLite metadata store (WAL mode) for sessions, approvals, and background jobs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any


class SessionLockRegistry:
    """Bounded per-session lock registry to prevent concurrent checkpoint races on the same thread_id."""

    def __init__(self, max_sessions: int = 2048):
        self._max_sessions = max_sessions
        self._guard = threading.Lock()
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()

    def for_session(self, session_id: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(session_id)
            if lock is not None:
                self._locks.move_to_end(session_id)
                return lock
            lock = threading.Lock()
            self._locks[session_id] = lock
            while len(self._locks) > self._max_sessions:
                self._locks.popitem(last=False)
            return lock


class ServiceMetaStore:
    """Encapsulates all service SQLite metadata tables behind an RLock and WAL mode."""

    def __init__(self, db_path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS service_sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_ledger (
                    session_id TEXT NOT NULL,
                    idem_key TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, idem_key)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS service_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    history_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def create_session(self, session_id: str, owner_id: str, tenant_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO service_sessions (session_id, owner_id, tenant_id, verified, created_at) VALUES (?, ?, ?, 0, ?)",
                (session_id, owner_id, tenant_id, time.time()),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT session_id, owner_id, tenant_id, verified FROM service_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

    def count_sessions(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM service_sessions").fetchone()[0])

    def get_approval_result(self, session_id: str, idem_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM approval_ledger WHERE session_id = ? AND idem_key = ?",
                (session_id, idem_key),
            ).fetchone()
        return None if row is None else json.loads(row["result_json"])

    def record_approval_result(self, session_id: str, idem_key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO approval_ledger (session_id, idem_key, result_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, idem_key, json.dumps(result, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def insert_job(self, job_id: str, job_type: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO service_jobs (job_id, job_type, status, result_json, error, history_json, created_at)
                VALUES (?, ?, 'pending', NULL, NULL, ?, ?)
                """,
                (job_id, job_type, json.dumps(["pending"]), now),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM service_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_payload(row)

    def update_job(self, job_id: str, status: str, result: dict[str, Any] | None, error: str | None) -> None:
        with self._lock:
            row = self._conn.execute("SELECT history_json FROM service_jobs WHERE job_id = ?", (job_id,)).fetchone()
            history = json.loads(row["history_json"]) if row else []
            history.append(status)
            self._conn.execute(
                """
                UPDATE service_jobs
                SET status = ?, result_json = ?, error = ?, history_json = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    None if result is None else json.dumps(result, ensure_ascii=False),
                    error,
                    json.dumps(history),
                    job_id,
                ),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM service_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else self._job_payload(row)

    @staticmethod
    def _job_payload(row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "result": result,
            "error": row["error"],
            "history": json.loads(row["history_json"]),
        }
