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

def message_usage(message: dict[str, Any]) -> dict[str, Any]:
    """Read provider usage. Cached tokens stay absent when the API did not report them."""
    usage = dict(message.get("usage") or {})
    raw_usage = ((message.get("raw_data") or {}).get("usage") or {})
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    if usage.get("prompt_tokens") is None and raw_usage.get("prompt_tokens") is not None:
        usage["prompt_tokens"] = raw_usage["prompt_tokens"]
    if usage.get("completion_tokens") is None and raw_usage.get("completion_tokens") is not None:
        usage["completion_tokens"] = raw_usage["completion_tokens"]
    if usage.get("cached_tokens") is None:
        details = raw_usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            usage["cached_tokens"] = details["cached_tokens"]
    return usage


def _require_reward(sim: dict[str, Any]) -> float:
    info = sim.get("reward_info")
    if not isinstance(info, dict) or info.get("reward") is None:
        raise RuntimeError(f"simulation {sim.get('task_id')} has no numeric reward")
    return float(info["reward"])


def rewards_for_tasks(results_json_path: Path, task_ids: list[str]) -> list[float]:
    """One mean reward per requested task. A missing task is an error, not a zero."""
    if not results_json_path.exists():
        raise FileNotFoundError(results_json_path)
    data = json.loads(results_json_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[float]] = {}
    for sim in data.get("simulations", []):
        grouped.setdefault(str(sim.get("task_id")), []).append(_require_reward(sim))
    missing = [task_id for task_id in task_ids if task_id not in grouped]
    if missing:
        raise RuntimeError(f"{results_json_path} is missing tasks: {missing[:8]}")
    return [sum(grouped[task_id]) / len(grouped[task_id]) for task_id in task_ids]


def summarize_simulation_file(results_json_path: Path, *, strict: bool = True) -> dict[str, float | int | None]:
    if not results_json_path.exists():
        raise FileNotFoundError(results_json_path)
    data = json.loads(results_json_path.read_text(encoding="utf-8"))
    sims = data.get("simulations", [])
    if not sims:
        raise RuntimeError(f"{results_json_path} has no simulations")

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
    cached_tokens: list[int] = []
    prompt_for_cache: list[int] = []
    message_counts: list[int] = []
    llm_call_counts: list[int] = []
    infra_errors = 0
    cache_observations = 0

    for sim in sims:
        reason = str(sim.get("termination_reason") or "")
        if reason in {"infrastructure_error", "unexpected_error"}:
            infra_errors += 1
        if strict and (reason in {"infrastructure_error", "unexpected_error"} or not isinstance(sim.get("reward_info"), dict)):
            raise RuntimeError(
                f"{results_json_path} task {sim.get('task_id')} ended with {reason or 'missing reward'}"
            )
        if not isinstance(sim.get("reward_info"), dict) or sim["reward_info"].get("reward") is None:
            continue
        rewards.append(float(sim["reward_info"]["reward"]))

        msgs = sim.get("messages") or []
        message_counts.append(len(msgs))
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        llm_call_counts.append(len(assistant_msgs))
        prompt_sum = 0
        completion_sum = 0
        for message in assistant_msgs:
            usage = message_usage(message)
            if usage.get("prompt_tokens") is None:
                continue
            prompt = int(usage["prompt_tokens"])
            prompt_sum += prompt
            completion_sum += int(usage.get("completion_tokens") or 0)
            if usage.get("cached_tokens") is not None:
                cached_tokens.append(int(usage["cached_tokens"]))
                prompt_for_cache.append(prompt)
                cache_observations += 1
        in_tokens.append(prompt_sum)
        out_tokens.append(completion_sum)

        tid = str(sim.get("task_id", ""))
        task_obj = tasks_map.get(tid, {})
        gold_acts = task_obj.get("evaluation_criteria", {}).get("actions", [])
        req_docs = task_obj.get("required_documents", [])
        nr, er = compute_action_recall(msgs, gold_acts)
        dr = compute_online_doc_recall(msgs, req_docs, all_doc_ids, title_to_id)
        act_name_recalls.append(nr)
        act_exact_recalls.append(er)
        doc_recalls.append(dr)

    if not rewards:
        raise RuntimeError(f"{results_json_path} has no scored simulations")
    n = len(rewards)
    avg_in = sum(in_tokens) / n
    avg_out = sum(out_tokens) / n
    # qwen3.8-flash card registered in the adapter: 0.8 RMB / M input, 2.7 RMB / M output.
    avg_cost_rmb = (avg_in * 0.8 + avg_out * 2.7) / 1e6
    cache_hit_rate = None
    if cache_observations and sum(prompt_for_cache) > 0:
        cache_hit_rate = sum(cached_tokens) / sum(prompt_for_cache)

    return {
        "n": n,
        "infra_errors": infra_errors,
        "pass_1": sum(rewards) / n,
        "action_name_recall": sum(act_name_recalls) / n,
        "action_exact_recall": sum(act_exact_recalls) / n,
        "online_doc_recall": sum(doc_recalls) / n,
        "avg_messages": sum(message_counts) / n,
        "avg_llm_calls": sum(llm_call_counts) / n,
        "avg_turns": sum(message_counts) / n,
        "avg_in_tokens": avg_in,
        "avg_out_tokens": avg_out,
        "avg_cached_tokens": (sum(cached_tokens) / cache_observations) if cache_observations else None,
        "cache_hit_rate": cache_hit_rate,
        "avg_cost_rmb": avg_cost_rmb,
    }
