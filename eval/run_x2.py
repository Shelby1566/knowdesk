"""X2 dev-30 factor comparison. Every cell uses max_steps=70.

Running this file calls Bailian. It does not fill missing tasks with reward 0.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gate import load_splits, report_ids
from eval.manifest import write_manifest
from eval.metrics import (
    compute_action_recall,
    compute_online_doc_recall,
    summarize_simulation_file,
)
from eval.tau_runner import DEFAULT_LLM_ARGS, DEFAULT_MODEL, run_tau
from eval.x2_protocol import PROTOCOL_MAX_STEPS, PROTOCOL_TRIALS, SEED, X2_CELLS, cell_by_key
from kb.loader import load_documents

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "RESULTS_X2.md"
SELECTED_PATH = Path(__file__).resolve().parent / "results" / "x2_selected.json"


def _rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def classify_failure(sim: dict, task: dict, doc_ids: set[str], title_to_id: dict[str, str]) -> str:
    messages = sim.get("messages") or []
    reward = float((sim.get("reward_info") or {}).get("reward"))
    if reward >= 1.0:
        return "0. 通过"
    if len(messages) <= 2:
        return "1. 评测或环境异常"
    required = task.get("required_documents") or []
    gold = [
        action for action in (task.get("evaluation_criteria") or {}).get("actions") or []
        if action.get("requestor") != "user"
    ]
    doc_recall = compute_online_doc_recall(messages, required, doc_ids, title_to_id)
    name_recall, exact_recall = compute_action_recall(messages, gold)
    if required and doc_recall < 0.30:
        return "2. 知识库检索漏召回"
    gold_hidden = any(action.get("name") == "call_discoverable_agent_tool" for action in gold)
    called_hidden = any(
        tool_call.get("name") == "call_discoverable_agent_tool"
        for message in messages if message.get("role") == "assistant"
        for tool_call in (message.get("tool_calls") or [])
    )
    if gold_hidden and not called_hidden:
        return "3. 隐藏工具未发现或未解锁"
    if name_recall < 0.70:
        return "4. 漏调关键业务工具"
    if exact_recall < name_recall:
        return "5. 工具参数填写错误"
    return "6. 子任务顺序或条款推理"


def _selection_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["pass_1"]),
        float(metrics["action_exact_recall"]),
        float(metrics["action_name_recall"]),
    )


def main() -> None:
    splits = load_splits()
    task_ids = report_ids(splits, "dev")
    if len(task_ids) != 30:
        raise RuntimeError(f"X2 tunes on dev 30, got {len(task_ids)}")
    rows = []
    for cell in X2_CELLS:
        run = run_tau(
            exp="x2",
            domain="banking_knowledge",
            agent=cell["agent"],
            retrieval=cell["retrieval"],
            prompt_version=cell["prompt_version"],
            guard_mode=cell["guard_mode"],
            compaction_mode=cell["compaction_mode"],
            task_ids=task_ids,
            max_steps=PROTOCOL_MAX_STEPS,
            num_trials=PROTOCOL_TRIALS,
            seed=SEED,
        )
        metrics = summarize_simulation_file(run["results_path"])
        if int(metrics["n"]) != len(task_ids):
            raise RuntimeError(f"{cell['key']} scored {metrics['n']} simulations, expected {len(task_ids)}")
        rows.append({"cell": cell, "run": run, "metrics": metrics})

    best = max(rows, key=lambda row: _selection_key(row["metrics"]))
    selected = {
        **cell_by_key(best["cell"]["key"]),
        "pass_1": best["metrics"]["pass_1"],
        "action_exact_recall": best["metrics"]["action_exact_recall"],
        "action_name_recall": best["metrics"]["action_name_recall"],
        "results_path": str(best["run"]["results_path"]),
        "max_steps": PROTOCOL_MAX_STEPS,
        "split_name": "dev30",
    }
    SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_PATH.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    documents = load_documents()
    doc_ids = {item.doc_id for item in documents}
    title_to_id = {item.title: item.doc_id for item in documents}
    from eval.metrics import load_tasks_map

    tasks = load_tasks_map()
    graph_row = next(row for row in rows if row["cell"]["key"] == "knowdesk_graph_hybrid")
    graph_data = json.loads(Path(graph_row["run"]["results_path"]).read_text(encoding="utf-8"))
    taxonomy: Counter[str] = Counter()
    for simulation in graph_data.get("simulations") or []:
        task = tasks.get(str(simulation.get("task_id")), {})
        taxonomy[classify_failure(simulation, task, doc_ids, title_to_id)] += 1

    lines = [
        f"# X2 dev 30 因子实验（max_steps={PROTOCOL_MAX_STEPS}，模型 `{DEFAULT_MODEL}`）",
        "",
        "五格配置只在 dev 30 上比较。平均消息条数是轨迹里的消息数，平均 LLM 调用是 assistant 消息数。",
        "cache_hit_rate = cached_tokens / prompt_tokens；供应商没返回 cached_tokens 时写 n/a。",
        f"按 Pass^1、精确参数召回、工具名召回选出的配置是 `{selected['key']}`。",
        "",
        "| 配置 | Pass^1 | 工具名召回 | 精确参数召回 | 在线文档召回 | 平均消息条数 | 平均 LLM 调用 | cache_hit_rate | 平均输入 token | 平均输出 token | 单单成本(元) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['cell']['label']} | {metrics['pass_1']:.4f} | {metrics['action_name_recall']:.4f} | "
            f"{metrics['action_exact_recall']:.4f} | {metrics['online_doc_recall']:.4f} | "
            f"{metrics['avg_messages']:.1f} | {metrics['avg_llm_calls']:.1f} | {_rate(metrics['cache_hit_rate'])} | "
            f"{metrics['avg_in_tokens']:.0f} | {metrics['avg_out_tokens']:.0f} | {metrics['avg_cost_rmb']:.4f} |"
        )
    lines.extend(["", "## knowdesk_graph + hybrid 失败归因", "", "| 类别 | 任务数 |", "|---|---:|"])
    for name, count in sorted(taxonomy.items()):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    write_manifest(
        "x2",
        model=DEFAULT_MODEL,
        llm_args=dict(DEFAULT_LLM_ARGS),
        seed=SEED,
        split_name="dev30",
        task_ids=task_ids,
        max_steps=PROTOCOL_MAX_STEPS,
        command="uv run --no-sync python eval/run_x2.py",
        metrics={row["cell"]["key"]: row["metrics"] for row in rows},
        extra={"selected": selected, "taxonomy_graph_hybrid": dict(taxonomy)},
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
