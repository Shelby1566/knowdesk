"""MCP tool backend with persistent JSON-RPC client session reuse.

Calls go through the official MCP Python SDK ``Client`` over the in-memory
JSON-RPC transport (``mode="legacy"``). The default ``mode="auto"`` path for an
in-process server skips JSON-RPC framing, so it is not used here.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

from mcp import Client

from knowdesk.mcp.server import SessionEnvironmentManager, create_knowdesk_mcp_server


def _tool_result_text(result: Any) -> str:
    content_list = getattr(result, "content", None)
    if content_list is None and isinstance(result, tuple) and len(result) > 0:
        content_list = result[0]
    if content_list is None and isinstance(result, list):
        content_list = result
    if content_list and len(content_list) > 0:
        first = content_list[0]
        return getattr(first, "text", str(first))
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return str(result)


class MCPToolBackend:
    """Plain 与 LangGraph 服务共用的 MCP 工具后端（复用持久 JSON-RPC 会话）。"""

    def __init__(
        self,
        env_manager: SessionEnvironmentManager | None = None,
        *,
        db_path: str | os.PathLike[str] | None = None,
        env_factory: Any = None,
        server_url: str | None = None,
    ):
        self.env_manager = env_manager
        self.server_url = server_url
        self.mcp_server = None
        if server_url is None:
            self.mcp_server, self.env_manager = create_knowdesk_mcp_server(
                env_manager, db_path=db_path, env_factory=env_factory
            )
        elif self.env_manager is None:
            self.env_manager = SessionEnvironmentManager(db_path=db_path, env_factory=env_factory)

        self._loop_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._active_client: Client | None = None
        self._client_async_lock: asyncio.Lock | None = None

    def _client(self) -> Client:
        if self.server_url:
            return Client(self.server_url)
        # legacy: InMemoryTransport + JSONRPCDispatcher. auto would skip JSON-RPC.
        return Client(self.mcp_server, mode="legacy")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            ready = threading.Event()

            def _runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._client_async_lock = asyncio.Lock()
                ready.set()
                loop.run_forever()

            self._thread = threading.Thread(target=_runner, name="knowdesk-mcp-loop", daemon=True)
            self._thread.start()
            ready.wait(timeout=5.0)
            assert self._loop is not None
            return self._loop

    async def _get_persistent_client(self) -> Client:
        if self._active_client is None:
            client = self._client()
            await client.__aenter__()
            self._active_client = client
        return self._active_client

    async def _reset_persistent_client(self) -> None:
        client = self._active_client
        self._active_client = None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass

    async def _invoke_on_bg_loop(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._client_async_lock is not None
        async with self._client_async_lock:
            try:
                client = await self._get_persistent_client()
                result = await client.call_tool(name, arguments)
            except Exception:
                await self._reset_persistent_client()
                client = await self._get_persistent_client()
                result = await client.call_tool(name, arguments)
        return _tool_result_text(result)

    def _submit(self, name: str, arguments: dict[str, Any]):
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(self._invoke_on_bg_loop(name, arguments), loop)

    def init_session(self, session_id: str, initial_state: Any = None) -> None:
        self.env_manager.get_or_create_env(session_id, initial_state=initial_state)

    async def call_tool_async(
        self, session_id: str, tool_name: str, arguments: dict[str, Any], requestor: str = "assistant"
    ) -> dict[str, Any]:
        fut = self._submit(
            "call_banking_tool",
            {
                "session_id": session_id,
                "tool_name": tool_name,
                "arguments_json": json.dumps(arguments, ensure_ascii=False),
                "requestor": requestor,
            },
        )
        raw = await asyncio.wrap_future(fut)
        return json.loads(raw)

    def call_tool(
        self, session_id: str, tool_name: str, arguments: dict[str, Any], requestor: str = "assistant"
    ) -> dict[str, Any]:
        fut = self._submit(
            "call_banking_tool",
            {
                "session_id": session_id,
                "tool_name": tool_name,
                "arguments_json": json.dumps(arguments, ensure_ascii=False),
                "requestor": requestor,
            },
        )
        raw = fut.result()
        return json.loads(raw)

    async def get_db_hash_async(self, session_id: str) -> str:
        fut = self._submit("get_test_db_hash", {"session_id": session_id})
        raw = await asyncio.wrap_future(fut)
        return raw.strip()

    def get_db_hash(self, session_id: str) -> str:
        fut = self._submit("get_test_db_hash", {"session_id": session_id})
        return fut.result().strip()

    def close(self) -> None:
        with self._loop_lock:
            loop = self._loop
            if loop is not None and loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(self._reset_persistent_client(), loop).result(timeout=3.0)
                except Exception:
                    pass
                loop.call_soon_threadsafe(loop.stop)
            self._loop = None
            self._thread = None
