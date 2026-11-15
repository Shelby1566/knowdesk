import json
from typing import Any
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from tau2.data_model.message import ToolCall
from tau2.registry import registry
from kb.hybrid_retriever import KnowDeskHybridRetriever

def _get_banking_env():
    import tau2.domains.banking_knowledge.retrieval as tau_ret
    tau_ret.RETRIEVAL_VARIANTS["alltools"] = tau_ret.RETRIEVAL_VARIANTS["bm25"]
    from tau2.domains.banking_knowledge.environment import get_environment
    return get_environment()

def get_banking_tasks():
    for attr in ("get_tasks", "get_tasks_fn"):
        if hasattr(registry, attr):
            res = getattr(registry, attr)("banking_knowledge")
            return res() if callable(res) else res
    for dict_attr in ("_tasks", "_task_loaders", "tasks"):
        d = getattr(registry, dict_attr, None)
        if isinstance(d, dict) and "banking_knowledge" in d:
            fn = d["banking_knowledge"]
            return fn() if callable(fn) else fn
    from tau2.domains.banking_knowledge.environment import get_tasks
    return get_tasks()

class SessionEnvironmentManager:
    """
    管理每个会话独立的 τ Environment 实例，并记录已执行的动作日志（Action Log）。
    当服务重启导致内存实例丢失时，通过重放 Action Log 重建 100% 一致的数据库状态。
    """
    def __init__(self):
        self._envs: dict[str, Any] = {}
        self._init_states: dict[str, Any] = {}
        self._action_logs: dict[str, list[dict[str, Any]]] = {}

    def get_or_create_env(self, session_id: str, initial_state: Any = None) -> Any:
        if initial_state is not None:
            self._init_states[session_id] = initial_state
        if session_id not in self._envs:
            env = _get_banking_env()
            init_st = self._init_states.get(session_id)
            if init_st is not None:
                env.set_state(
                    initialization_data=init_st.initialization_data,
                    initialization_actions=init_st.initialization_actions,
                    message_history=[],
                )
            for entry in self._action_logs.get(session_id, []):
                tc = ToolCall(
                    id=entry["id"], name=entry["name"], arguments=entry["arguments"], requestor=entry["requestor"]
                )
                env.get_response(tc)
            self._envs[session_id] = env
            self._action_logs.setdefault(session_id, [])
        return self._envs[session_id]

    def simulate_restart_clear_memory(self) -> None:
        self._envs.clear()

    def execute_tool(
        self, session_id: str, tool_name: str, arguments: dict[str, Any], requestor: str = "assistant", initial_state: Any = None
    ) -> dict[str, Any]:
        env = self.get_or_create_env(session_id, initial_state=initial_state)
        if tool_name == "KB_search":
            q = str(arguments.get("query", ""))
            k = int(arguments.get("top_k", 10))
            hits = KnowDeskHybridRetriever.get_instance().search(q, k=k)
            return {"content": json.dumps(hits, ensure_ascii=False), "error": False}

        call_id = f"mcp_{len(self._action_logs[session_id])}"
        tc = ToolCall(id=call_id, name=tool_name, arguments=arguments, requestor=requestor)
        resp = env.get_response(tc)
        self._action_logs[session_id].append(
            {"id": call_id, "name": tool_name, "arguments": arguments, "requestor": requestor}
        )
        return {
            "content": str(getattr(resp, "content", "")),
            "error": bool(getattr(resp, "error", False)),
        }

    def get_session_db_hash(self, session_id: str) -> str:
        env = self.get_or_create_env(session_id)
        return str(env.get_db_hash())

def create_knowdesk_mcp_server(env_manager: SessionEnvironmentManager | None = None) -> tuple[MCPServer, SessionEnvironmentManager]:
    mgr = env_manager or SessionEnvironmentManager()
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
