"""Chinese retail, first 50 tasks, with the NL assertion judge pointed at Bailian.

DB match and NL assertion are reported separately from the combined reward.
Infrastructure errors are written into the report and then raised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.manifest import write_manifest
from eval.tau_runner import (
    DEFAULT_LLM_ARGS,
    DEFAULT_MODEL,
    INFRA_REASONS,
    TAU_ROOT,
    run_tau,
    signed_results_path,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "RESULTS_X9_RETAIL_ZH.md"
TASKS_PATH = TAU_ROOT / "data" / "tau2" / "domains" / "retail" / "tasks.json"
RETAIL_MAX_STEPS = 50
SEED = 42


def retail_task_ids() -> list[str]:
    if not TASKS_PATH.exists():
        raise FileNotFoundError(TASKS_PATH)
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    ids = [str(task["id"]) for task in tasks[:50]]
    if len(ids) != 50:
        raise RuntimeError(f"retail tasks file has only {len(ids)} tasks in the first slice")
    if len(set(ids)) != 50:
        raise RuntimeError("the first 50 retail task ids are not unique")
    return ids


def _component(simulation: dict[str, Any], key: str) -> float | None:
    info = simulation.get("reward_info") or {}
    if key == "DB":
        db_check = info.get("db_check") or {}
        if db_check.get("db_reward") is not None:
            return float(db_check["db_reward"])
    breakdown = info.get("reward_breakdown") or {}
    if key in breakdown and breakdown[key] is not None:
        return float(breakdown[key])
    return None


def _guard_refusal(simulation: dict[str, Any]) -> bool:
    for message in simulation.get("messages") or []:
        content = str(message.get("content") or "")
        if "guard_blocked" in content or "R02_" in content or "R03_" in content or "R04_" in content or "R05_" in content:
            return True
    return False


def _summarize(path: Path, task_ids: list[str]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(item.get("task_id")): item for item in data.get("simulations") or []}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    rewards = []
    db_scores = []
    nl_scores = []
    buckets: dict[str, list[str]] = {
        "infrastructure_error": [],
        "missing_reward": [],
        "max_steps": [],
        "db_mismatch": [],
        "nl_fail": [],
        "guard_refusal": [],
        "success": [],
        "missing_task": missing,
    }
    for task_id in task_ids:
        if task_id not in by_id:
            continue
        simulation = by_id[task_id]
        reason = str(simulation.get("termination_reason") or "")
        if reason in INFRA_REASONS:
            buckets["infrastructure_error"].append(task_id)
        if "max" in reason or reason in {"too_many_steps", "max_steps"}:
            buckets["max_steps"].append(task_id)
        reward = (simulation.get("reward_info") or {}).get("reward")
        if reward is None:
            buckets["missing_reward"].append(task_id)
            continue
        reward = float(reward)
        rewards.append(reward)
        if reward >= 1.0:
            buckets["success"].append(task_id)
        db_reward = _component(simulation, "DB")
        if db_reward is None:
            buckets.setdefault("db_missing", []).append(task_id)
        else:
            db_scores.append(db_reward)
            if db_reward < 1.0:
                buckets["db_mismatch"].append(task_id)
        nl_reward = _component(simulation, "NL_ASSERTION")
        if nl_reward is not None:
            nl_scores.append(nl_reward)
            if nl_reward < 1.0:
                buckets["nl_fail"].append(task_id)
        if _guard_refusal(simulation):
            buckets["guard_refusal"].append(task_id)
    return {
        "n": len(task_ids),
        "scored": len(rewards),
        "infra_errors": len(buckets["infrastructure_error"]),
        "pass_1": (sum(rewards) / len(rewards)) if rewards else None,
        "db_match": (sum(db_scores) / len(db_scores)) if db_scores else None,
        "db_n": len(db_scores),
        "nl_assertion": (sum(nl_scores) / len(nl_scores)) if nl_scores else None,
        "nl_n": len(nl_scores),
        "buckets": buckets,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _relation(candidate: float, baseline: float) -> str:
    if candidate > baseline:
        return "高于"
    if candidate < baseline:
        return "低于"
    return "等于"


def main() -> None:
    task_ids = retail_task_ids()
    cells = (
        {
            "key": "tau_retail",
            "label": "τ llm_agent",
            "agent": "llm_agent",
            "prompt_version": "tau",
            "guard_mode": "0",
        },
        {
            "key": "knowdesk_retail",
            "label": "knowdesk_graph",
            "agent": "knowdesk_graph",
            "prompt_version": "v2",
            "guard_mode": "1",
        },
    )
    summaries = []
    for cell in cells:
        kwargs = dict(
            domain="retail",
            agent=cell["agent"],
            retrieval="builtin",
            prompt_version=cell["prompt_version"],
            guard_mode=cell["guard_mode"],
            compaction_mode="0",
            task_ids=task_ids,
            max_steps=RETAIL_MAX_STEPS,
            num_trials=1,
            language="zh",
        )
        try:
            run = run_tau(exp="x9", seed=SEED, nl_judge_model=DEFAULT_MODEL, **kwargs)
        except RuntimeError as exc:
            text = str(exc)
            if "infrastructure errors" not in text and "has no reward" not in text:
                raise
            run = {"results_path": signed_results_path(**kwargs)}
        summary = _summarize(run["results_path"], task_ids)
        summary["label"] = cell["label"]
        summary["key"] = cell["key"]
        summaries.append(summary)
    baseline, candidate = summaries
    lines = [
        "# X9 中文 retail 前 50 单",
        "",
        f"任务号取零售任务文件的前 50 个，max_steps={RETAIL_MAX_STEPS}，语言 `zh`，检索 `builtin`。",
        "两边的 NL assertion 裁判都显式指向百炼，避免 τ 默认的 OpenAI 裁判造成基础设施错误。",
        "DB Match 与 NL Assertion 分开统计。没有该项得分的任务不补 0。",
        "",
        "| 配置 | 有奖励的任务 | Pass^1（综合） | DB Match | DB 任务数 | NL Assertion | NL 任务数 | Infra Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['scored']} | {_fmt(summary['pass_1'])} | "
            f"{_fmt(summary['db_match'])} | {summary['db_n']} | {_fmt(summary['nl_assertion'])} | "
            f"{summary['nl_n']} | {summary['infra_errors']} |"
        )
    lines.extend(["", "## 对照", ""])
    if baseline["pass_1"] is None or candidate["pass_1"] is None:
        lines.append("有配置没有综合奖励，不能比较 Pass^1。")
    else:
        lines.append(
            f"knowdesk_graph 的综合 Pass^1 `{candidate['pass_1']:.4f}` "
            f"{_relation(candidate['pass_1'], baseline['pass_1'])} "
            f"τ 基线 `{baseline['pass_1']:.4f}`。"
        )
    lines.extend(["", "## 失败归因", ""])
    for summary in summaries:
        lines.append(f"### {summary['label']}")
        lines.append("")
        for name, ids in summary["buckets"].items():
            shown = ", ".join(ids[:15])
            suffix = "" if len(ids) <= 15 else " ..."
            lines.append(f"- {name}: {len(ids)}" + (f" ({shown}{suffix})" if ids else ""))
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    write_manifest(
        "x9",
        model=DEFAULT_MODEL,
        llm_args=dict(DEFAULT_LLM_ARGS),
        seed=SEED,
        split_name="retail_zh_first50",
        task_ids=task_ids,
        max_steps=RETAIL_MAX_STEPS,
        command="uv run --no-sync python eval/run_x9_retail_zh.py",
        metrics={
            summary["key"]: {key: value for key, value in summary.items() if key != "buckets"}
            | {"bucket_counts": {name: len(ids) for name, ids in summary["buckets"].items()}}
            for summary in summaries
        },
        extra={"nl_judge_model": DEFAULT_MODEL, "retrieval": "builtin", "language": "zh"},
    )
    infra = sum(summary["infra_errors"] for summary in summaries)
    missing = sum(len(summary["buckets"]["missing_reward"]) for summary in summaries)
    missing_tasks = sum(len(summary["buckets"]["missing_task"]) for summary in summaries)
    print(f"wrote {REPORT_PATH}")
    if infra or missing or missing_tasks or any(summary["scored"] != 50 for summary in summaries):
        raise RuntimeError(
            f"retail run is not clean: infra={infra}, missing_reward={missing}, "
            f"missing_task={missing_tasks}. The report was written before this error."
        )


if __name__ == "__main__":
    main()
