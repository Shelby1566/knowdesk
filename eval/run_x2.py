"""X2 dev-30 comparison. Three cells, max_steps=90.

Running this file calls Bailian. It does not fill missing tasks with reward 0.
The five-cell list in eval/x2_protocol.py stays available for X3; this entrypoint
runs only the three release cells.
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
    rewards_for_tasks,
    summarize_simulation_file,
)
from eval.pass_k import paired_bootstrap_diff
from eval.tau_runner import DEFAULT_LLM_ARGS, DEFAULT_MODEL, run_tau
from eval.x2_protocol import PROTOCOL_TRIALS, SEED, cell_by_key
from kb.loader import load_documents

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "RESULTS_X2.md"
SELECTED_PATH = Path(__file__).resolve().parent / "results" / "x2_selected.json"
# 60 stopped 18/30 graph runs while later writes were still pending. Guard and
# section truncation were not the argument errors. 90 keeps the same three cells.
RELEASE_MAX_STEPS = 90
RELEASE_CELL_KEYS = (
    "llm_agent_builtin",
    "knowdesk_plain_hybrid",
    "knowdesk_graph_hybrid",
)
RELEASE_LABELS = {
    "llm_agent_builtin": "τ 原生基线 (llm_agent + 内置 BM25)",
    "knowdesk_plain_hybrid": "KnowDesk-Plain (knowdesk_plain + 自研混合检索 + Fail-Closed Guard)",
    "knowdesk_graph_hybrid": "KnowDesk-LangGraph (knowdesk_graph + 自研混合检索 + Fail-Closed Guard)",
}


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
    for key in RELEASE_CELL_KEYS:
        cell = cell_by_key(key)
        cell["label"] = RELEASE_LABELS[key]
        run = run_tau(
            exp="x2",
            domain="banking_knowledge",
            agent=cell["agent"],
            retrieval=cell["retrieval"],
            prompt_version=cell["prompt_version"],
            guard_mode=cell["guard_mode"],
            compaction_mode=cell["compaction_mode"],
            task_ids=task_ids,
            max_steps=RELEASE_MAX_STEPS,
            num_trials=PROTOCOL_TRIALS,
            seed=SEED,
        )
        metrics = summarize_simulation_file(run["results_path"])
        if int(metrics["n"]) != len(task_ids):
            raise RuntimeError(f"{cell['key']} scored {metrics['n']} simulations, expected {len(task_ids)}")
        scores = rewards_for_tasks(run["results_path"], task_ids)
        rows.append({"cell": cell, "run": run, "metrics": metrics, "scores": scores})

    best = max(rows, key=lambda row: _selection_key(row["metrics"]))
    selected = {
        **cell_by_key(best["cell"]["key"]),
        "label": best["cell"]["label"],
        "pass_1": best["metrics"]["pass_1"],
        "action_exact_recall": best["metrics"]["action_exact_recall"],
        "action_name_recall": best["metrics"]["action_name_recall"],
        "results_path": str(best["run"]["results_path"]),
        "max_steps": RELEASE_MAX_STEPS,
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

    baseline_scores = rows[0]["scores"]
    lines = [
        f"# X2 dev 30（max_steps={RELEASE_MAX_STEPS}，temperature=0，思考关闭，模型 `{DEFAULT_MODEL}`）",
        "",
        "三组使用同一批 dev 30、同一个 max_steps。Pass^1 是任务奖励均值。",
        "步数上限是 90。上一轮 60 步时，knowdesk_graph 有 18/30 条在上限结束。检索、Guard 和模型没有改。",
        "配对 Bootstrap 是候选减 τ 基线，10000 次、种子 42、95% 区间。基线行没有对照差。",
        f"按 Pass^1、精确参数召回、工具名召回选出的配置是 `{selected['key']}`。",
        "",
        "| 配置 | Pass^1 | 工具名召回 | 精确参数召回 | 在线文档召回 | 平均输入 token | 相对基线 Pass^1 差 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row["metrics"]
        if row["cell"]["key"] == "llm_agent_builtin":
            interval = "n/a（对照）"
        else:
            diff = paired_bootstrap_diff(baseline_scores, row["scores"], seed=SEED)
            interval = f"[{diff['ci_95_low']:.4f}, {diff['ci_95_high']:.4f}] {diff['verdict']}"
        lines.append(
            f"| {row['cell']['label']} | {metrics['pass_1']:.4f} | {metrics['action_name_recall']:.4f} | "
            f"{metrics['action_exact_recall']:.4f} | {metrics['online_doc_recall']:.4f} | "
            f"{metrics['avg_in_tokens']:.0f} | {interval} |"
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
        max_steps=RELEASE_MAX_STEPS,
        command="uv run --no-sync python eval/run_x2.py",
        metrics={row["cell"]["key"]: row["metrics"] for row in rows},
        extra={"selected": selected, "taxonomy_graph_hybrid": dict(taxonomy)},
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
