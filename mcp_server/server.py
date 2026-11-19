"""KnowDesk banking MCP server.

Action logs live in SQLite so a new process can replay them onto a fresh τ
environment. A missing banking_knowledge environment is a hard error.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

try:
    from tau2.data_model.message import ToolCall
    from tau2.registry import registry

    _HAS_TAU2 = True
except ImportError:
    _HAS_TAU2 = False
    from pydantic import BaseModel

    class ToolCall(BaseModel):
        id: str
        name: str
        arguments: dict[str, Any] | str = {}
        requestor: str = "assistant"

    registry = None

from kb.hybrid_retriever import KnowDeskHybridRetriever

EnvFactory = Callable[[], Any]


def _load_tau_environment() -> Any:
    import tau2.domains.banking_knowledge.retrieval as tau_ret

    tau_ret.RETRIEVAL_VARIANTS["alltools"] = tau_ret.RETRIEVAL_VARIANTS["bm25"]
    from tau2.domains.banking_knowledge.environment import get_environment

    return get_environment()


def _get_banking_env() -> Any:
    """Load the real τ banking_knowledge environment. Never substitutes a mock."""
    if not _HAS_TAU2:
        raise RuntimeError("tau2 banking_knowledge environment is not installed")
    try:
        return _load_tau_environment()
    except Exception as exc:
        raise RuntimeError("failed to load the tau banking_knowledge environment") from exc


def get_banking_tasks():
    for attr in ("get_tasks", "get_tasks_fn"):
        if hasattr(registry, attr):
            res = getattr(registry, attr)("banking_knowledge")
            return res() if callable(res) else res
    for dict_attr in ("_tasks", "_task_loaders", "tasks"):
        loaded = getattr(registry, dict_attr, None)
        if isinstance(loaded, dict) and "banking_knowledge" in loaded:
            fn = loaded["banking_knowledge"]
            return fn() if callable(fn) else fn
    from tau2.domains.banking_knowledge.environment import get_tasks

    return get_tasks()


def _default_db_path() -> str:
    return os.getenv("KNOWDESK_DB_PATH", "./data/knowdesk.sqlite")


def _prepare_sqlite_path(db_path: str | os.PathLike[str] | None) -> str:
    path = str(db_path or _default_db_path())
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_initial_state(state: Any) -> str | None:
    if state is None:
        return None
    if hasattr(state, "model_dump"):
        payload = state.model_dump(mode="json")
        cls = f"{type(state).__module__}:{type(state).__qualname__}"
        return json.dumps({"cls": cls, "data": payload}, ensure_ascii=False)
    return json.dumps({"cls": "", "data": state}, ensure_ascii=False)


def _restore_initial_state(raw: str | None) -> Any:
    if not raw:
        return None
    obj = json.loads(raw)
    cls_path = obj.get("cls") or ""
    data = obj.get("data")
    if not cls_path:
        return data
    module_name, _, qualname = cls_path.rpartition(":")
    module = __import__(module_name, fromlist=[qualname.split(".")[0]])
    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls(**data)


class SessionEnvironmentManager:
    """每个会话一份 τ Environment。动作日志在 SQLite，新实例按日志重放恢复数据库。"""

    def __init__(self, db_path: str | os.PathLike[str] | None = None, env_factory: EnvFactory | None = None):
        self.db_path = _prepare_sqlite_path(db_path)
        self._env_factory = env_factory or _get_banking_env
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_action_logs (
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                requestor TEXT NOT NULL,
                PRIMARY KEY (session_id, seq)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_session_init (
                session_id TEXT PRIMARY KEY,
                init_json TEXT
            )
            """
        )
        self._conn.commit()
        self._envs: dict[str, Any] = {}

    def _save_init(self, session_id: str, initial_state: Any) -> None:
        self._conn.execute(
            """
            INSERT INTO mcp_session_init (session_id, init_json) VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET init_json = excluded.init_json
            """,
            (session_id, _serialize_initial_state(initial_state)),
        )
        self._conn.commit()

    def _load_init(self, session_id: str) -> Any:
        row = self._conn.execute(
            "SELECT init_json FROM mcp_session_init WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return _restore_initial_state(row["init_json"])

    def _load_logs(self, session_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT seq, call_id, tool_name, args_json, requestor
                FROM mcp_action_logs
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (session_id,),
            )
        )

    def _next_seq(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM mcp_action_logs WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])

    def get_or_create_env(self, session_id: str, initial_state: Any = None) -> Any:
        with self._lock:
            cached = self._envs.get(session_id)
            if cached is not None:
                return cached
            if initial_state is not None:
                self._save_init(session_id, initial_state)
            else:
                initial_state = self._load_init(session_id)
            env = self._env_factory()
            if initial_state is not None:
                env.set_state(
                    initialization_data=initial_state.initialization_data,
                    initialization_actions=initial_state.initialization_actions,
                    message_history=[],
                )
            for entry in self._load_logs(session_id):
                arguments = json.loads(entry["args_json"]) if entry["args_json"] else {}
                tc = ToolCall(
                    id=entry["call_id"],
                    name=entry["tool_name"],
                    arguments=arguments,
                    requestor=entry["requestor"],
                )
                env.get_response(tc)
            self._envs[session_id] = env
            return env

    def simulate_restart_clear_memory(self) -> None:
        """丢掉内存中的环境对象。下一次访问从 SQLite 动作日志重放。"""
        with self._lock:
            self._envs.clear()

    def execute_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        requestor: str = "assistant",
        initial_state: Any = None,
    ) -> dict[str, Any]:
        env = self.get_or_create_env(session_id, initial_state=initial_state)
        if tool_name == "KB_search":
            query = str(arguments.get("query", ""))
            top_k = int(arguments.get("top_k", 10))
            hits = KnowDeskHybridRetriever.get_instance().search_sections(query, k=top_k)
            return {"content": json.dumps(hits, ensure_ascii=False), "error": False}

        with self._lock:
            seq = self._next_seq(session_id)
            call_id = f"mcp_{seq}"
            tc = ToolCall(id=call_id, name=tool_name, arguments=arguments, requestor=requestor)
            resp = env.get_response(tc)
            self._conn.execute(
                """
                INSERT INTO mcp_action_logs (session_id, seq, call_id, tool_name, args_json, requestor)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    seq,
                    call_id,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    requestor,
                ),
            )
            self._conn.commit()
        return {
            "content": str(getattr(resp, "content", "")),
            "error": bool(getattr(resp, "error", False)),
        }

    def get_session_db_hash(self, session_id: str) -> str:
        env = self.get_or_create_env(session_id)
        return str(env.get_db_hash())


def create_knowdesk_mcp_server(
    env_manager: SessionEnvironmentManager | None = None,
    *,
    db_path: str | os.PathLike[str] | None = None,
    env_factory: EnvFactory | None = None,
) -> tuple[MCPServer, SessionEnvironmentManager]:
    mgr = env_manager or SessionEnvironmentManager(db_path=db_path, env_factory=env_factory)
    mcp = MCPServer("knowdesk-banking-mcp")

    @mcp.tool()
    def call_banking_tool(
        session_id: str, tool_name: str, arguments_json: str = "{}", requestor: str = "assistant"
    ) -> str:
        args = json.loads(arguments_json) if arguments_json.strip() else {}
        res = mgr.execute_tool(session_id=session_id, tool_name=tool_name, arguments=args, requestor=requestor)
        return json.dumps(res, ensure_ascii=False)

    @mcp.tool()
    def get_test_db_hash(session_id: str) -> str:
        return mgr.get_session_db_hash(session_id)

    return mcp, mgr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KnowDesk banking MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    server, _manager = create_knowdesk_mcp_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
        return
    server.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
