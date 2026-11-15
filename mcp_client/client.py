import asyncio
import concurrent.futures
import json
from typing import Any
from mcp_server.server import SessionEnvironmentManager, create_knowdesk_mcp_server

def _run_coroutine_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        return asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

class MCPToolBackend:
    """M36: 基于 MCP Python SDK v2 的工具执行后端（plain 与 LangGraph 服务模式共用）"""
    def __init__(self, env_manager: SessionEnvironmentManager | None = None):
        self.mcp_server, self.env_manager = create_knowdesk_mcp_server(env_manager)

    async def _invoke_mcp_tool(self, name: str, arguments: dict[str, Any]) -> str:
        res = await self.mcp_server.call_tool(name, arguments)
        content_list = getattr(res, "content", None)
        if content_list is None and isinstance(res, tuple) and len(res) > 0:
            content_list = res[0]
        if content_list is None and isinstance(res, list):
            content_list = res
        if content_list and len(content_list) > 0:
            first = content_list[0]
            return getattr(first, "text", str(first))
        return str(res)

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
