import hashlib
import json
import random
import sqlite3
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from knowdesk.tools.schemas import StructuredToolError, is_tool_result_success


class RetryableToolException(Exception):
    def __init__(self, message: str, error_type: str = "timeout"):
        super().__init__(message)
        self.error_type = error_type


class InvalidArgumentToolException(Exception):
    pass


def compute_idempotency_key(session_id: str, tool_name: str, args: dict[str, Any]) -> str:
    canonical_args = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{session_id}||{tool_name}||{canonical_args}".encode("utf-8")).hexdigest()


class SqliteIdempotencyStore:
    """跨进程重启仍生效的写操作幂等账本。"""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_ledger (
                session_id TEXT NOT NULL,
                idem_key TEXT NOT NULL,
                tool_name TEXT,
                result_json TEXT,
                created_at TEXT,
                PRIMARY KEY (session_id, idem_key)
            )
            """
        )
        self._conn.commit()

    def get(self, session_id: str, idem_key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM idempotency_ledger WHERE session_id = ? AND idem_key = ?",
                (session_id, idem_key),
            ).fetchone()
        return None if row is None else row[0]

    def put(self, session_id: str, idem_key: str, tool_name: str, result_json: str) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO idempotency_ledger
                    (session_id, idem_key, tool_name, result_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, idem_key, tool_name, result_json, created_at),
            )
            self._conn.commit()


class ReliableToolExecutor:
    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng_fn: Callable[[float, float], float] = random.uniform,
        max_cache_size: int = 1000,
        idempotency_db_path: str | Path | None = None,
        block_consecutive_read_duplicates: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.sleep_fn = sleep_fn
        self.rng_fn = rng_fn
        self.max_cache_size = max_cache_size
        self.block_consecutive_read_duplicates = block_consecutive_read_duplicates
        self.idempotency_store: OrderedDict[str, str] = OrderedDict()
        self.ledger = SqliteIdempotencyStore(idempotency_db_path) if idempotency_db_path else None
        self.last_call_tracker: OrderedDict[str, tuple[str, int]] = OrderedDict()
        self.duplicate_intercept_count: int = 0
        self._state_lock = threading.Lock()
        self._session_locks: OrderedDict[str, threading.Lock] = OrderedDict()

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._state_lock:
            lock = self._session_locks.get(session_id)
            if lock is not None:
                self._session_locks.move_to_end(session_id)
                return lock
            lock = threading.Lock()
            self._session_locks[session_id] = lock
            while len(self._session_locks) > self.max_cache_size:
                self._session_locks.popitem(last=False)
            return lock

    def _remember(self, idem_key: str, out_str: str) -> None:
        if idem_key in self.idempotency_store:
            self.idempotency_store.move_to_end(idem_key)
            self.idempotency_store[idem_key] = out_str
            return
        while len(self.idempotency_store) >= self.max_cache_size:
            self.idempotency_store.popitem(last=False)
        self.idempotency_store[idem_key] = out_str

    def _record_last_call(self, session_id: str, idem_key: str, count: int) -> None:
        if session_id in self.last_call_tracker:
            self.last_call_tracker.move_to_end(session_id)
        self.last_call_tracker[session_id] = (idem_key, count)
        while len(self.last_call_tracker) > self.max_cache_size:
            self.last_call_tracker.popitem(last=False)

    def execute(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        func: Callable[[dict[str, Any]], Any],
        is_write: bool = False,
        allow_idempotent_replay: bool = True,
        idem_key: str | None = None,
    ) -> str:
        # 仅对同一 session 的写操作加细粒度会话锁，避免不同 session 或只读调用被全局锁串行阻塞。
        if is_write:
            with self._get_session_lock(session_id):
                return self._execute_impl(
                    session_id, tool_name, args, func, is_write, allow_idempotent_replay, idem_key
                )
        return self._execute_impl(
            session_id, tool_name, args, func, is_write, allow_idempotent_replay, idem_key
        )

    def _execute_impl(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        func: Callable[[dict[str, Any]], Any],
        is_write: bool,
        allow_idempotent_replay: bool,
        idem_key: str | None = None,
    ) -> str:
        idem_key = idem_key or compute_idempotency_key(session_id, tool_name, args)
        if is_write and allow_idempotent_replay:
            if self.ledger is not None:
                cached = self.ledger.get(session_id, idem_key)
                if cached is not None:
                    return cached
            else:
                with self._state_lock:
                    if idem_key in self.idempotency_store:
                        self.idempotency_store.move_to_end(idem_key)
                        return self.idempotency_store[idem_key]

        with self._state_lock:
            prev_key, count = self.last_call_tracker.get(session_id, ("", 0))
            if idem_key == prev_key and not is_write:
                count += 1
                self._record_last_call(session_id, idem_key, count)
                if self.block_consecutive_read_duplicates and count >= 2:
                    self.duplicate_intercept_count += 1
                    return StructuredToolError(
                        error_type="duplicate_call",
                        message=f"Tool '{tool_name}' called repeatedly with identical arguments.",
                        retryable=False,
                        hint="Use previous result or change parameters.",
                    ).to_tool_message()
            else:
                self._record_last_call(session_id, idem_key, 1)

        attempt = 0
        while True:
            try:
                raw_out = func(args)
                out_str = raw_out if isinstance(raw_out, str) else json.dumps(raw_out, ensure_ascii=False)
                # Business failures must not be replayed. A later retry has to hit the tool again.
                if is_write and is_tool_result_success(out_str):
                    if self.ledger is not None:
                        self.ledger.put(session_id, idem_key, tool_name, out_str)
                    else:
                        with self._state_lock:
                            self._remember(idem_key, out_str)
                return out_str
            except InvalidArgumentToolException as e:
                return StructuredToolError(
                    error_type="invalid_arguments", message=str(e), retryable=False
                ).to_tool_message()
            except RetryableToolException as e:
                if attempt >= self.max_retries:
                    return StructuredToolError(
                        error_type="timeout" if e.error_type == "timeout" else "upstream_error",
                        message=f"Failed after {attempt + 1} attempts: {e}",
                        retryable=True,
                    ).to_tool_message()
                cap = min(self.max_delay, self.base_delay * (2 ** attempt))
                self.sleep_fn(self.rng_fn(0.0, cap))
                attempt += 1
            except Exception as e:
                return StructuredToolError(
                    error_type="upstream_error", message=f"Unexpected tool error: {e}", retryable=False
                ).to_tool_message()
