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

from adapters.tau2_adapter import register_knowdesk_components
from tau2.cli import main as tau2_cli_main

SIM_ROOT = Path(__file__).resolve().parents[2] / "tau2-bench/data/simulations"
REPORT_X9_PATH = Path(__file__).resolve().parents[1] / "RESULTS_X9_RETAIL_ZH.md"
MODEL_NAME = os.getenv("KNOWDESK_MODEL", "dashscope/qwen3.8-flash")

def run_retail_zh(save_name: str, agent_name: str, num_tasks: int = 10) -> Path:
    res_file = SIM_ROOT / save_name / "results.json"
    if res_file.exists():
        try:
            sims = json.loads(res_file.read_text(encoding="utf-8")).get("simulations", [])
            if len(sims) >= num_tasks:
                return res_file
        except Exception:
            pass

    register_knowdesk_components()
    llm_args = json.dumps({"temperature": 0, "extra_body": {"enable_thinking": False}})
    argv = [
        "tau2", "run",
        "--domain", "retail",
        "--language", "zh",
        "--agent", agent_name,
        "--agent-llm", MODEL_NAME,
        "--user-llm", MODEL_NAME,
        "--agent-llm-args", llm_args,
        "--user-llm-args", llm_args,
        "--num-tasks", str(num_tasks),
        "--max-steps", "50",
        "--max-concurrency", "6",
        "--save-to", save_name,
    ]
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
    num_tasks = 50 if "--full50" in sys.argv else 10
    p_base = run_retail_zh(f"x9_retail_zh_base_n{num_tasks}", "llm_agent", num_tasks)
    p_graph = run_retail_zh(f"x9_retail_zh_graph_n{num_tasks}", "knowdesk_graph", num_tasks)

    def _calc(p: Path):
        sims = json.loads(p.read_text(encoding="utf-8")).get("simulations", [])
        n = len(sims)
        pass_db = sum(float((s.get("reward_info") or {}).get("reward", 0.0)) for s in sims) / n if n else 0.0
        turns = sum(len(s.get("messages", [])) for s in sims) / n if n else 0.0
        return n, pass_db, turns

    n1, r1, t1 = _calc(p_base)
    n2, r2, t2 = _calc(p_graph)
    md = "\n".join([
        f"# X9 中文第二域 (`retail --language zh`) 迁移评测结果 (n={n2}, 模型: `{MODEL_NAME}`)\n",
        "| 配置 | DB 终态通过率 (Pass^1) | 平均交互步数 |",
        "|---|---:|---:|",
        f"| 1. τ原生基线 (`llm_agent`) | {r1:.4f} | {t1:.1f} |",
        f"| 2. KnowDesk-LangGraph (`knowdesk_graph` + Guard) | {r2:.4f} | {t2:.1f} |\n",
    ])
    REPORT_X9_PATH.write_text(md, encoding="utf-8")
    print("\n" + "=" * 70)
    print(md)

if __name__ == "__main__":
    main()
