"""X3 reads the five signed X2 runs and computes paired bootstrap differences.

It does not launch τ and it does not fill a missing task with reward 0.
The regression-12 rows are a slice of the same dev-30 runs, not a second protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gate import load_splits, report_ids, tunable_ids
from eval.manifest import write_manifest
from eval.metrics import rewards_for_tasks, summarize_simulation_file
from eval.pass_k import paired_bootstrap_diff
from eval.tau_runner import DEFAULT_LLM_ARGS, DEFAULT_MODEL, signed_results_path
from eval.x2_protocol import (
    BASELINE_KEY,
    PROTOCOL_MAX_STEPS,
    PROTOCOL_TRIALS,
    SEED,
    X2_CELLS,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/benchmarks/RESULTS_X3_ABLATION.md"


def _results_for(cell: dict[str, str], task_ids: list[str]) -> Path:
    return signed_results_path(
        domain="banking_knowledge",
        agent=cell["agent"],
        retrieval=cell["retrieval"],
        prompt_version=cell["prompt_version"],
        guard_mode=cell["guard_mode"],
        compaction_mode=cell["compaction_mode"],
        task_ids=task_ids,
        max_steps=PROTOCOL_MAX_STEPS,
        num_trials=PROTOCOL_TRIALS,
        language=None,
    )


def _compare(name: str, baseline: list[float], candidate: list[float]) -> dict:
    diff = paired_bootstrap_diff(baseline, candidate, seed=SEED)
    return {"config": name, "pass_1": sum(candidate) / len(candidate), **diff}


def main() -> None:
    splits = load_splits()
    dev_ids = report_ids(splits, "dev")
    regression_ids = tunable_ids(splits, "regression")
    if len(dev_ids) != 30:
        raise RuntimeError(f"expected dev 30, got {len(dev_ids)}")
    paths = {cell["key"]: _results_for(cell, dev_ids) for cell in X2_CELLS}
    rewards = {key: rewards_for_tasks(path, dev_ids) for key, path in paths.items()}
    baseline = rewards[BASELINE_KEY]
    dev_rows = [
        _compare(cell["label"], baseline, rewards[cell["key"]])
        for cell in X2_CELLS
        if cell["key"] != BASELINE_KEY
    ]
    index = {task_id: position for position, task_id in enumerate(dev_ids)}
    regression_rows = []
    for cell in X2_CELLS:
        if cell["key"] == BASELINE_KEY:
            continue
        base_slice = [baseline[index[task_id]] for task_id in regression_ids]
        cand_slice = [rewards[cell["key"]][index[task_id]] for task_id in regression_ids]
        regression_rows.append(_compare(cell["label"], base_slice, cand_slice))

    def table(title: str, rows: list[dict]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "对照是同一批任务上的 `llm_agent + builtin`。判定使用预先登记的 delta=0.05，没有在 test 上调整。",
            "",
            "| 配置 | Pass^1 | 平均差 | 95% CI 下界 | 95% CI 上界 | 判定 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['config']} | {row['pass_1']:.4f} | {row['mean_diff']:.4f} | "
                f"{row['ci_95_low']:.4f} | {row['ci_95_high']:.4f} | {row['verdict']} |"
            )
        lines.append("")
        return lines

    lines = [
        "# X3 dev 30 配对比较",
        "",
        f"五格结果都来自 max_steps={PROTOCOL_MAX_STEPS} 的 X2 签名目录。本脚本没有重新跑 τ。",
        "regression 子集整段落在 dev 内，只作为诊断，不另选配置。",
        "",
    ]
    lines.extend(table(f"dev {len(dev_ids)}", dev_rows))
    lines.extend(table(f"regression {len(regression_ids)}（dev 的子集）", regression_rows))
    summaries = {key: summarize_simulation_file(path) for key, path in paths.items()}
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    signature = json.loads((paths[BASELINE_KEY].parent / "knowdesk_signature.json").read_text(encoding="utf-8"))
    write_manifest(
        "x3",
        model=str(signature.get("model") or DEFAULT_MODEL),
        llm_args=signature.get("llm_args") or dict(DEFAULT_LLM_ARGS),
        seed=int(signature.get("seed") or SEED),
        split_name="dev30_from_x2",
        task_ids={"dev": dev_ids, "regression": regression_ids},
        max_steps=PROTOCOL_MAX_STEPS,
        command="uv run --no-sync python eval/run_x3_regression.py",
        metrics={"dev": dev_rows, "regression": regression_rows, "summaries": summaries},
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
