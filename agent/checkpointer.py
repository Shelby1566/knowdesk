import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def _import_saver(module: str, attr: str) -> Any:
    """可选后端缺少驱动时抛出 ImportError，禁止静默退回 SQLite。"""
    try:
        imported = __import__(module, fromlist=[attr])
    except ImportError as exc:
        raise ImportError(
            f"{module} is not installed; refusing to fall back to sqlite"
        ) from exc
    return getattr(imported, attr)


def create_checkpointer(backend: str | None = None, db_url_or_path: str | Path = ":memory:") -> Any:
    """
    Checkpointer 工厂：
    - memory: InMemorySaver
    - sqlite: SqliteSaver
    - postgres / redis: 仅在对应驱动已安装时构造；未安装则抛出 ImportError
    """
    mode = (backend or os.getenv("KNOWDESK_CHECKPOINTER_BACKEND", "sqlite")).lower()
    if mode == "memory":
        return InMemorySaver()
    if mode == "postgres":
        postgres_saver = _import_saver("langgraph.checkpoint.postgres", "PostgresSaver")
        return postgres_saver.from_conn_string(str(db_url_or_path))
    if mode == "redis":
        redis_saver = _import_saver("langgraph.checkpoint.redis", "RedisSaver")
        return redis_saver.from_conn_string(str(db_url_or_path))
    if mode == "sqlite":
        conn = sqlite3.connect(str(db_url_or_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    raise ValueError(
        f"Unknown checkpointer backend '{mode}'. Expected memory, sqlite, postgres, or redis."
    )
