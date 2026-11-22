"""X8 reports Pass^1 and Pass^3 on the frozen test 67 split.

The agent and retrieval come from eval/results/x2_selected.json. Compaction
comes from eval/results/x5_selected.json. Either missing file raises.
This script does not choose hyperparameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gate import load_splits, report_ids
from eval.manifest import write_manifest
from eval.pass_k import bootstrap_ci, dataset_pass_k, paired_bootstrap_diff
from eval.tau_runner import DEFAULT_LLM_ARGS, DEFAULT_MODEL, run_tau
from eval.x2_protocol import BASELINE_KEY, PROTOCOL_MAX_STEPS, SEED, cell_by_key

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/benchmarks/RESULTS_X8_TEST67.md"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
X2_SELECTED = RESULTS_DIR / "x2_selected.json"
X5_SELECTED = RESULTS_DIR / "x5_selected.json"
NUM_TRIALS = 3


def _load_selection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Finish the dev-30 X2 and X5 runs before scoring test 67."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return data


def _trial_rewards(path: Path, task_ids: list[str]) -> dict[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped = {task_id: [] for task_id in task_ids}
    for simulation in data.get("simulations") or []:
        task_id = str(simulation.get("task_id"))
        if task_id not in grouped:
            continue
        reward = (simulation.get("reward_info") or {}).get("reward")
        if reward is None:
            raise RuntimeError(f"{path} task {task_id} has no reward")
        grouped[task_id].append(float(reward))
    short = [task_id for task_id, rewards in grouped.items() if len(rewards) < NUM_TRIALS]
    if short:
        raise RuntimeError(f"{path} is missing trials for {short[:8]}")
    return {task_id: rewards[:NUM_TRIALS] for task_id, rewards in grouped.items()}


def _pack(label: str, grouped: dict[str, list[float]], order: list[str]) -> dict:
    means = [sum(grouped[task_id]) / NUM_TRIALS for task_id in order]
    successes = [
        (NUM_TRIALS, sum(reward >= 1.0 for reward in grouped[task_id]))
        for task_id in order
    ]
    pass_1 = sum(means) / len(means)
    pass_3 = dataset_pass_k(successes, 3)
    pass_1_mean, pass_1_lo, pass_1_hi = bootstrap_ci(means, seed=SEED)
    per_task_pass_3 = [1.0 if count == NUM_TRIALS else 0.0 for _n, count in successes]
    _pass_3_mean, pass_3_lo, pass_3_hi = bootstrap_ci(per_task_pass_3, seed=SEED)
    return {
        "label": label,
        "pass_1": pass_1,
        "pass_1_ci_low": pass_1_lo,
        "pass_1_ci_high": pass_1_hi,
        "pass_3": pass_3,
        "pass_3_ci_low": pass_3_lo,
        "pass_3_ci_high": pass_3_hi,
        "means": means,
        "bootstrap_pass_1": pass_1_mean,
    }


def main() -> None:
    x2 = _load_selection(X2_SELECTED)
    x5 = _load_selection(X5_SELECTED)
    for field in ("agent", "retrieval", "prompt_version", "guard_mode"):
        if field not in x2:
            raise RuntimeError(f"x2_selected.json is missing {field}")
    compaction = str(x5.get("compaction_mode"))
    if compaction not in {"0", "1"}:
        raise RuntimeError("x5_selected.json compaction_mode must be 0 or 1")
    splits = load_splits()
    task_ids = report_ids(splits, "test")
    if len(task_ids) != 67:
        raise RuntimeError(f"expected test 67, got {len(task_ids)}")
    baseline_cell = cell_by_key(BASELINE_KEY)
    candidate = {
        "agent": x2["agent"],
        "retrieval": x2["retrieval"],
        "prompt_version": x2["prompt_version"],
        "guard_mode": x2["guard_mode"],
        "compaction_mode": compaction if x2["agent"] != "llm_agent" else "0",
    }
    runs = {}
    for label, cell in (("baseline", baseline_cell), ("candidate", candidate)):
        runs[label] = run_tau(
            exp="x8",
            domain="banking_knowledge",
            agent=cell["agent"],
            retrieval=cell["retrieval"],
            prompt_version=cell["prompt_version"],
            guard_mode=cell["guard_mode"],
            compaction_mode=cell["compaction_mode"],
            task_ids=task_ids,
            max_steps=PROTOCOL_MAX_STEPS,
            num_trials=NUM_TRIALS,
            seed=SEED,
            allow_held_out=True,
        )
    baseline = _pack("τ 原生基线", _trial_rewards(runs["baseline"]["results_path"], task_ids), task_ids)
    candidate_pack = _pack(
        f"{candidate['agent']} + {candidate['retrieval']} compaction={candidate['compaction_mode']}",
        _trial_rewards(runs["candidate"]["results_path"], task_ids),
        task_ids,
    )
    diff = paired_bootstrap_diff(baseline["means"], candidate_pack["means"], seed=SEED)
    lines = [
        "# X8 test 67 可靠性",
        "",
        f"每个任务独立 {NUM_TRIALS} 次，max_steps={PROTOCOL_MAX_STEPS}，种子 {SEED}。",
        "配置来自 dev 上已经冻结的 X2 / X5 选择文件。本脚本没有在 test 上改参数。",
        f"候选配置：`{json.dumps(candidate, ensure_ascii=False)}`。",
        "",
        "| 配置 | Pass^1 | Pass^1 95% CI | Pass^3 | Pass^3 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    for row in (baseline, candidate_pack):
        lines.append(
            f"| {row['label']} | {row['pass_1']:.4f} | [{row['pass_1_ci_low']:.4f}, {row['pass_1_ci_high']:.4f}] | "
            f"{row['pass_3']:.4f} | [{row['pass_3_ci_low']:.4f}, {row['pass_3_ci_high']:.4f}] |"
        )
    lines.extend([
        "",
        f"候选减基线的配对平均差 `{diff['mean_diff']:.4f}`，"
        f"95% CI `[{diff['ci_95_low']:.4f}, {diff['ci_95_high']:.4f}]`，判定 `{diff['verdict']}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    public_baseline = {key: value for key, value in baseline.items() if key != "means"}
    public_candidate = {key: value for key, value in candidate_pack.items() if key != "means"}
    write_manifest(
        "x8",
        model=DEFAULT_MODEL,
        llm_args=dict(DEFAULT_LLM_ARGS),
        seed=SEED,
        split_name="test67",
        task_ids=task_ids,
        max_steps=PROTOCOL_MAX_STEPS,
        command="uv run --no-sync python eval/run_x8_test67.py",
        metrics={"baseline": public_baseline, "candidate": public_candidate, "paired": diff},
        extra={"candidate": candidate, "num_trials": NUM_TRIALS},
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
