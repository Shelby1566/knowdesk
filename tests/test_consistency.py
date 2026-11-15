from tau2.data_model.message import ToolCall
from mcp_server.server import _get_banking_env, get_banking_tasks
from mcp_client.client import MCPToolBackend

def test_m36_dual_path_db_hash_consistency_and_negative_case():
    tasks = get_banking_tasks()[:10]
    backend = MCPToolBackend()

    for idx, task in enumerate(tasks):
        env_a = _get_banking_env()
        if task.initial_state is not None:
            env_a.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=[],
            )

        sid = f"consist_session_{idx}"
        backend.init_session(sid, initial_state=task.initial_state)

        gold_actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
        for step_i, act in enumerate(gold_actions):
            req = getattr(act, "requestor", "assistant") or "assistant"
            tc = ToolCall(id=f"a_{step_i}", name=act.name, arguments=act.arguments, requestor=req)
            resp_a = env_a.get_response(tc)
            resp_b = backend.call_tool(sid, act.name, act.arguments, requestor=req)
            assert str(getattr(resp_a, "content", "")) == resp_b["content"]

        # 模拟重启清空内存后，验证通过 Action Log 重放恢复的 DB 哈希依然与路径 A 100% 一致
        backend.env_manager.simulate_restart_clear_memory()
        assert str(env_a.get_db_hash()) == backend.get_db_hash(sid)

    # 反例测试：寻找一个改变 DB 状态的任务，路径 B 不执行写动作，断言哈希必定不同
    for task in tasks:
        env_full = _get_banking_env()
        env_empty = _get_banking_env()
        if task.initial_state is not None:
            for e in (env_full, env_empty):
                e.set_state(
                    initialization_data=task.initial_state.initialization_data,
                    initialization_actions=task.initial_state.initialization_actions,
                    message_history=[],
                )
        for step_i, act in enumerate(task.evaluation_criteria.actions):
            req = getattr(act, "requestor", "assistant") or "assistant"
            env_full.get_response(ToolCall(id=f"f_{step_i}", name=act.name, arguments=act.arguments, requestor=req))
        if str(env_full.get_db_hash()) != str(env_empty.get_db_hash()):
            backend.init_session("neg_test", initial_state=task.initial_state)
            assert str(env_full.get_db_hash()) != backend.get_db_hash("neg_test")
            break
