import builtins
import json
import os
import sys
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / "tau2-bench/.env", override=True)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
builtins.input = lambda *args, **kwargs: "y"

from adapters.tau2_adapter import register_knowdesk_components, patch_retrieval_mode
from eval.gate import load_splits, require_tunable, tunable_ids
from eval.metrics import summarize_simulation_file, load_tasks_map, compute_action_recall, compute_online_doc_recall
from kb.loader import load_documents
from tau2.cli import main as tau2_cli_main

SIM_ROOT = Path(__file__).resolve().parents[2] / "tau2-bench/data/simulations"
REPORT_X2_PATH = Path(__file__).resolve().parents[1] / "RESULTS_X2.md"
TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "docs/FAILURE_TAXONOMY.md"
MODEL_NAME = os.getenv("KNOWDESK_MODEL", "dashscope/qwen3.8-flash")

def run_one_config(save_name: str, agent_name: str, retrieval_mode: str, task_ids: list[str]) -> Path:
    res_file = SIM_ROOT / save_name / "results.json"
    if res_file.exists():
        try:
            existing = json.loads(res_file.read_text(encoding="utf-8")).get("simulations", [])
            if len(existing) >= len(task_ids):
                print(f"⏩ [{save_name}] 已完成 ({len(existing)}/{len(task_ids)})，直接读取缓存结果。")
                return res_file
        except Exception:
            pass

    os.environ["KNOWDESK_RETRIEVAL"] = retrieval_mode
    register_knowdesk_components()
    patch_retrieval_mode(retrieval_mode)

    llm_args = json.dumps({"temperature": 0, "extra_body": {"enable_thinking": False}})
    argv = [
        "tau2", "run",
        "--domain", "banking_knowledge",
        "--retrieval-config", "bm25",
        "--agent", agent_name,
        "--agent-llm", MODEL_NAME,
        "--user-llm", MODEL_NAME,
        "--agent-llm-args", llm_args,
        "--user-llm-args", llm_args,
        "--max-steps", "40",
        "--max-concurrency", "6",
        "--save-to", save_name,
        "--task-ids", *task_ids,
    ]
    print(f"\n🚀 开始运行配置 [{save_name}]: agent={agent_name}, retrieval={retrieval_mode}, tasks={len(task_ids)}")
    old_argv = sys.argv
    try:
        sys.argv = argv
        tau2_cli_main()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"⚠️ CLI exit code: {e.code}")
    finally:
        sys.argv = old_argv
    return res_file

def classify_failure(sim: dict, task_obj: dict, all_doc_ids: set[str], title_to_id: dict[str, str]) -> str:
    msgs = sim.get("messages", [])
    if len(msgs) <= 2:
        return "1. 评测/环境侧异常 (ENV_OR_SIMULATOR_ISSUE)"

    req_docs = task_obj.get("required_documents", [])
    gold_acts = [a for a in task_obj.get("evaluation_criteria", {}).get("actions", []) if a.get("requestor") != "user"]
    doc_rec = compute_online_doc_recall(msgs, req_docs, all_doc_ids, title_to_id)
    name_rec, exact_rec = compute_action_recall(msgs, gold_acts)

    if req_docs and doc_rec < 0.30:
        return "2. 知识库检索漏召回 (RETRIEVAL_MISS)"

    gold_has_disc = any(a.get("name") == "call_discoverable_agent_tool" for a in gold_acts)
    called_disc = any(
        tc.get("name") == "call_discoverable_agent_tool"
        for m in msgs if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    )
    if gold_has_disc and not called_disc:
        return "3. 隐藏工具未发现/未解锁 (DISCOVERABLE_TOOL_MISS)"
    if name_rec < 0.70:
        return "4. 漏调关键业务工具 (MISSING_REQUIRED_ACTION)"
    if exact_rec < name_rec:
        return "5. 工具参数填写错误 (WRONG_TOOL_ARGUMENTS)"
    return "6. 子任务顺序或复杂条款推理偏差 (SUBTASK_ORDER_OR_REASONING)"

def main():
    splits = load_splits()
    use_full = "--full" in sys.argv
    task_ids = tunable_ids(splits, "dev" if use_full else "regression10")
    require_tunable(task_ids, splits)
    tag = "dev30" if use_full else "dev10_lite"

    configs = [
        (f"x2_{tag}_1_base_bm25", "1. τ原生基线 (LLMAgent + 内置BM25)", "llm_agent", "builtin"),
        (f"x2_{tag}_2_plain_bm25", "2. KnowDesk-Plain (手写循环+Guard) + 内置BM25", "knowdesk_plain", "builtin"),
        (f"x2_{tag}_3_plain_hybrid", "3. KnowDesk-Plain (手写循环+Guard) + 自研混合检索", "knowdesk_plain", "hybrid"),
        (f"x2_{tag}_4_graph_hybrid", "4. KnowDesk-LangGraph (主图+Guard) + 自研混合检索", "knowdesk_graph", "hybrid"),
    ]

    summary_rows = []
    last_res_path = None
    for save_name, label, agent_name, ret_mode in configs:
        res_path = run_one_config(save_name, agent_name, ret_mode, task_ids)
        last_res_path = res_path
        m = summarize_simulation_file(res_path)
        summary_rows.append((label, m))

    lines = [
        f"# X2 端到端工单办理 2×2 对照实验结果 (数据集: {tag}, n={len(task_ids)}, 模型: `{MODEL_NAME}`)\n",
        "| 实验配置 | Pass^1 (DB终态) | Action Recall (工具名) | Action Recall (精确参数) | 在线 Doc Recall | 平均轮数 | 平均输入Token | 单单成本(¥) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, m in summary_rows:
        if not m:
            continue
        lines.append(
            f"| {label} | {m['pass_1']:.4f} | {m['action_name_recall']:.4f} | {m['action_exact_recall']:.4f} | "
            f"{m['online_doc_recall']:.4f} | {m['avg_turns']:.1f} | {m['avg_in_tokens']:.0f} | ¥{m['avg_cost_rmb']:.3f} |"
        )

    tasks_map = load_tasks_map()
    docs = load_documents()
    all_doc_ids = {d.doc_id for d in docs}
    title_to_id = {d.title: d.doc_id for d in docs}

    if last_res_path and last_res_path.exists():
        data = json.loads(last_res_path.read_text(encoding="utf-8"))
        failed_sims = [s for s in data.get("simulations", []) if float((s.get("reward_info") or {}).get("reward", 0.0)) < 1.0]
        cat_counter: Counter[str] = Counter()
        examples_by_cat: dict[str, list[str]] = {}
        for s in failed_sims:
            tid = str(s.get("task_id", ""))
            cat = classify_failure(s, tasks_map.get(tid, {}), all_doc_ids, title_to_id)
            cat_counter[cat] += 1
            examples_by_cat.setdefault(cat, []).append(tid)

        lines.append(f"\n## M20 失败轨迹归因统计 (基于配置 4 失败工单，共 {len(failed_sims)} 单)\n")
        lines.append("| 失败主因类别 | 数量 | 占失败总数比例 | 典型 Task ID |")
        lines.append("|---|---:|---:|---|")
        for cat, cnt in sorted(cat_counter.items()):
            ratio = (cnt / len(failed_sims) * 100.0) if failed_sims else 0.0
            ex_str = ", ".join(examples_by_cat[cat][:3])
            lines.append(f"| {cat} | {cnt} | {ratio:.1f}% | `{ex_str}` |")

        TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAXONOMY_PATH.write_text("\n".join(lines), encoding="utf-8")

    report_text = "\n".join(lines) + "\n"
    REPORT_X2_PATH.write_text(report_text, encoding="utf-8")
    print("\n" + "=" * 75)
    print(report_text)
    print(f"✅ X2 报告已保存至: {REPORT_X2_PATH}")

if __name__ == "__main__":
    main()
