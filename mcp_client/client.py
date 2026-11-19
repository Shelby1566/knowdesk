"""MCP tool backend.

Calls go through the official MCP Python SDK ``Client`` over the in-memory
JSON-RPC transport (``mode="legacy"``). The default ``mode="auto"`` path for an
in-process server skips JSON-RPC framing, so it is not used here.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any

from mcp import Client

from mcp_server.server import SessionEnvironmentManager, create_knowdesk_mcp_server


def _run_coroutine_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


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
    """Plain 与 LangGraph 服务共用的 MCP 工具后端。"""

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

    def _client(self) -> Client:
        if self.server_url:
            return Client(self.server_url)
        # legacy: InMemoryTransport + JSONRPCDispatcher. auto would skip JSON-RPC.
        return Client(self.mcp_server, mode="legacy")

    async def _invoke_mcp_tool(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._client() as client:
            result = await client.call_tool(name, arguments)
        return _tool_result_text(result)

    def init_session(self, session_id: str, initial_state: Any = None) -> None:
        self.env_manager.get_or_create_env(session_id, initial_state=initial_state)

    async def call_tool_async(
        self, session_id: str, tool_name: str, arguments: dict[str, Any], requestor: str = "assistant"
    ) -> dict[str, Any]:
        raw = await self._invoke_mcp_tool(
            "call_banking_tool",
            {
                "session_id": session_id,
                "tool_name": tool_name,
                "arguments_json": json.dumps(arguments, ensure_ascii=False),
                "requestor": requestor,
            },
        )
        return json.loads(raw)

    def call_tool(
        self, session_id: str, tool_name: str, arguments: dict[str, Any], requestor: str = "assistant"
    ) -> dict[str, Any]:
        return _run_coroutine_sync(self.call_tool_async(session_id, tool_name, arguments, requestor=requestor))

    async def get_db_hash_async(self, session_id: str) -> str:
        raw = await self._invoke_mcp_tool("get_test_db_hash", {"session_id": session_id})
        return raw.strip()

    def get_db_hash(self, session_id: str) -> str:
        return _run_coroutine_sync(self.get_db_hash_async(session_id))
