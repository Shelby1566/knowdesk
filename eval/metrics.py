import json
from pathlib import Path
from typing import Any
from agent.guard import unwrap_tool_call
from kb.loader import DEFAULT_TASKS_PATH, load_documents

def load_tasks_map(tasks_path: Path = DEFAULT_TASKS_PATH) -> dict[str, dict[str, Any]]:
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    return {str(t["id"]): t for t in raw}

def compute_action_recall(sim_messages: list[dict[str, Any]], gold_actions: list[dict[str, Any]]) -> tuple[float, float]:
    """
    计算 Action Recall（剥离 call_discoverable_agent_tool 外壳后按内层真实工具名与参数比对）。
    返回: (tool_name_recall, exact_arg_recall)
    """
    agent_gold = [a for a in gold_actions if a.get("requestor") != "user"]
    if not agent_gold:
        return 1.0, 1.0

    called_names: set[str] = set()
    called_exact: set[tuple[str, str]] = set()

    for m in sim_messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            real_name, real_args, _ = unwrap_tool_call(tc.get("name", ""), tc.get("arguments", {}))
            called_names.add(real_name)
            canon = json.dumps(real_args, sort_keys=True, ensure_ascii=False)
            called_exact.add((real_name, canon))

    name_hits = 0
    exact_hits = 0
    for ga in agent_gold:
        g_name, g_args, _ = unwrap_tool_call(ga.get("name", ""), ga.get("arguments", {}))
        g_canon = json.dumps(g_args, sort_keys=True, ensure_ascii=False)
        if g_name in called_names:
            name_hits += 1
        if (g_name, g_canon) in called_exact:
            exact_hits += 1

    return name_hits / len(agent_gold), exact_hits / len(agent_gold)

def compute_online_doc_recall(
    sim_messages: list[dict[str, Any]],
    required_docs: list[str],
    all_doc_ids: set[str],
    title_to_id: dict[str, str],
) -> float:
    """从轨迹的所有 ToolMessage 返回文本中提取召回过的政策文档 ID，计算在线 Document Recall"""
    if not required_docs:
        return 1.0
    gold_set = set(str(d) for d in required_docs)
    retrieved: set[str] = set()

    tool_blobs: list[str] = []
    for m in sim_messages:
        if m.get("role") == "tool" and m.get("content"):
            tool_blobs.append(str(m["content"]))
        elif m.get("tool_messages"):
            for tm in m["tool_messages"]:
                if tm.get("content"):
                    tool_blobs.append(str(tm["content"]))

    full_text = "\n".join(tool_blobs)
    if not full_text:
        return 0.0

    for did in gold_set:
        if did in full_text:
            retrieved.add(did)
    for title, did in title_to_id.items():
        if did in gold_set and len(title) > 6 and title in full_text:
            retrieved.add(did)

    return len(retrieved & gold_set) / len(gold_set)

def summarize_simulation_file(results_json_path: Path) -> dict[str, float]:
    data = json.loads(results_json_path.read_text(encoding="utf-8"))
    sims = data.get("simulations", [])
    if not sims:
        return {}

    tasks_map = load_tasks_map()
    docs = load_documents()
    all_doc_ids = {d.doc_id for d in docs}
    title_to_id = {d.title: d.doc_id for d in docs}

    rewards: list[float] = []
    act_name_recalls: list[float] = []
    act_exact_recalls: list[float] = []
    doc_recalls: list[float] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    turns_list: list[int] = []

    for sim in sims:
        tid = str(sim.get("task_id", ""))
        rew = float((sim.get("reward_info") or {}).get("reward", 0.0))
        rewards.append(rew)

        msgs = sim.get("messages", [])
        turns_list.append(len(msgs))
        in_tokens.append(sum((m.get("usage") or {}).get("prompt_tokens", 0) for m in msgs))
        out_tokens.append(sum((m.get("usage") or {}).get("completion_tokens", 0) for m in msgs))

        task_obj = tasks_map.get(tid, {})
        gold_acts = task_obj.get("evaluation_criteria", {}).get("actions", [])
        req_docs = task_obj.get("required_documents", [])

        nr, er = compute_action_recall(msgs, gold_acts)
        dr = compute_online_doc_recall(msgs, req_docs, all_doc_ids, title_to_id)
        act_name_recalls.append(nr)
        act_exact_recalls.append(er)
        doc_recalls.append(dr)

    n = len(sims)
    avg_in = sum(in_tokens) / n
    avg_out = sum(out_tokens) / n
    # 按 qwen3.8-flash 定价 (输入 0.8元/M, 输出 2.7元/M) 估算单任务人民币成本
    avg_cost_rmb = (avg_in * 0.8 + avg_out * 2.7) / 1e6

    return {
        "n": n,
        "pass_1": sum(rewards) / n,
        "action_name_recall": sum(act_name_recalls) / n,
        "action_exact_recall": sum(act_exact_recalls) / n,
        "online_doc_recall": sum(doc_recalls) / n,
        "avg_turns": sum(turns_list) / n,
        "avg_in_tokens": avg_in,
        "avg_out_tokens": avg_out,
        "avg_cost_rmb": avg_cost_rmb,
    }
