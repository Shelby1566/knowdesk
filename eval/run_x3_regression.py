import builtins
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / "tau2-bench/.env", override=True)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
builtins.input = lambda *args, **kwargs: "y"

from adapters.tau2_adapter import register_knowdesk_components, patch_retrieval_mode
from eval.metrics import summarize_simulation_file
from eval.pass_k import paired_bootstrap_diff
from tau2.cli import main as tau2_cli_main

SPLITS_PATH = Path(__file__).resolve().parent / "splits.json"
SIM_ROOT = Path(__file__).resolve().parents[2] / "tau2-bench/data/simulations"
REPORT_X3_PATH = Path(__file__).resolve().parents[1] / "RESULTS_X3_ABLATION.md"
MODEL_NAME = os.getenv("KNOWDESK_MODEL", "dashscope/qwen3.8-flash")

def get_task_rewards(res_path: Path, task_ids: list[str]) -> list[float]:
    data = json.loads(res_path.read_text(encoding="utf-8"))
    rmap = {str(s["task_id"]): float((s.get("reward_info") or {}).get("reward", 0.0)) for s in data.get("simulations", [])}
    return [rmap.get(tid, 0.0) for tid in task_ids]

def run_v2_config(save_name: str, agent_name: str, guard_on: str, task_ids: list[str]) -> Path:
    res_file = SIM_ROOT / save_name / "results.json"
    if res_file.exists():
        try:
            existing = json.loads(res_file.read_text(encoding="utf-8")).get("simulations", [])
            if len(existing) >= len(task_ids):
                print(f"⏩ [{save_name}] 已存在缓存 ({len(existing)}/{len(task_ids)})。")
                return res_file
        except Exception:
            pass

    os.environ["KNOWDESK_RETRIEVAL"] = "hybrid"
    os.environ["KNOWDESK_PROMPT_VERSION"] = "v2"
    os.environ["KNOWDESK_GUARD"] = guard_on
    register_knowdesk_components()
    patch_retrieval_mode("hybrid")

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
        "--max-steps", "70",
        "--max-concurrency", "6",
        "--save-to", save_name,
        "--task-ids", *task_ids,
    ]
    print(f"\n🚀 运行 X3 配置 [{save_name}]: prompt=v2.0, guard={guard_on}, max_steps=70")
    old_argv = sys.argv
    try:
        sys.argv = argv
        tau2_cli_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    return res_file

def main():
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    task_ids = splits["regression_ids"][:10]

    base_path = SIM_ROOT / "x2_dev10_lite_1_base_bm25/results.json"
    v1_graph_path = SIM_ROOT / "x2_dev10_lite_4_graph_hybrid/results.json"
    v2_graph_path = run_v2_config("x3_dev10_v2_graph_hybrid", "knowdesk_graph", "1", task_ids)

    m_base = summarize_simulation_file(base_path)
    m_v1 = summarize_simulation_file(v1_graph_path)
    m_v2 = summarize_simulation_file(v2_graph_path)

    r_base = get_task_rewards(base_path, task_ids)
    r_v1 = get_task_rewards(v1_graph_path, task_ids)
    r_v2 = get_task_rewards(v2_graph_path, task_ids)

    gate_vs_base = paired_bootstrap_diff(r_base, r_v2, seed=42)
    gate_vs_v1 = paired_bootstrap_diff(r_v1, r_v2, seed=42)

    lines = [
        f"# X3 失败归因驱动迭代与 M19 回归门禁报告 (n={len(task_ids)}, 模型: `{MODEL_NAME}`)\n",
        "| 版本与配置 | Pass^1 (DB终态) | Action Recall (工具名) | Action Recall (精确参数) | 在线 Doc Recall | 平均轮数 | 单单成本(¥) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 1. τ原生基线 (v0, 内置BM25) | {m_base['pass_1']:.4f} | {m_base['action_name_recall']:.4f} | {m_base['action_exact_recall']:.4f} | {m_base['online_doc_recall']:.4f} | {m_base['avg_turns']:.1f} | ¥{m_base['avg_cost_rmb']:.3f} |",
        f"| 2. KnowDesk-LangGraph v1.0 (M20归因前) | {m_v1['pass_1']:.4f} | {m_v1['action_name_recall']:.4f} | {m_v1['action_exact_recall']:.4f} | {m_v1['online_doc_recall']:.4f} | {m_v1['avg_turns']:.1f} | ¥{m_v1['avg_cost_rmb']:.3f} |",
        f"| 3. KnowDesk-LangGraph v2.0 (M20归因修复后) | {m_v2['pass_1']:.4f} | {m_v2['action_name_recall']:.4f} | {m_v2['action_exact_recall']:.4f} | {m_v2['online_doc_recall']:.4f} | {m_v2['avg_turns']:.1f} | ¥{m_v2['avg_cost_rmb']:.3f} |",
        "\n## M19 配对 Bootstrap (10,000次重抽样) 回归门禁结论\n",
        f"- **v2.0 相对 τ原生基线**: $\\bar{{\\Delta}} = {gate_vs_base['mean_diff']:+.4f}$, 95% CI = `[{gate_vs_base['ci_95_low']:+.4f}, {gate_vs_base['ci_95_high']:+.4f}]` $\\rightarrow$ 门禁判定: **`{gate_vs_base['verdict']}`**",
        f"- **v2.0 相对 v1.0**: $\\bar{{\\Delta}} = {gate_vs_v1['mean_diff']:+.4f}$, 95% CI = `[{gate_vs_v1['ci_95_low']:+.4f}, {gate_vs_v1['ci_95_high']:+.4f}]` $\\rightarrow$ 门禁判定: **`{gate_vs_v1['verdict']}`**\n",
    ]
    report = "\n".join(lines)
    REPORT_X3_PATH.write_text(report, encoding="utf-8")
    print("\n" + "=" * 75)
    print(report)

if __name__ == "__main__":
    main()
