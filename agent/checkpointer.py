import os
import sqlite3
from pathlib import Path
from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

def create_checkpointer(backend: str | None = None, db_url_or_path: str | Path = ":memory:") -> Any:
    """
    分布式 Checkpointer 抽象工厂：
    - 'memory': InMemorySaver（单测与无状态快速评测）
    - 'sqlite': SqliteSaver（单机持久化与跨进程重启恢复）
    - 'postgres' / 'redis': 生产环境 K8s 多副本部署时对接外部高可用集群（未安装驱动时优雅回退至 SQLite）
    """
    mode = (backend or os.getenv("KNOWDESK_CHECKPOINTER_BACKEND", "sqlite")).lower()
    if mode == "memory":
        return InMemorySaver()
    if mode == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
            return PostgresSaver.from_conn_string(str(db_url_or_path))
        except Exception:
            pass
    if mode == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver  # type: ignore
            return RedisSaver.from_conn_string(str(db_url_or_path))
        except Exception:
            pass
    conn = sqlite3.connect(str(db_url_or_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
