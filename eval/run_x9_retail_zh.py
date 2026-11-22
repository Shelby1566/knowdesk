"""Chinese retail, first 30 tasks, deterministic DB Match only.

The OpenAI NL assertion judge is removed from the reward basis for this process,
and communicate-info uses substring matching. Neither path calls OpenAI.
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
    run_tau,
    signed_results_path,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/benchmarks/RESULTS_X9_RETAIL_ZH.md"
RETAIL_N = 30
RETAIL_MAX_STEPS = 50
SEED = 42
DB_ONLY_PROTOCOL = "dbonly"


def _without_nl_assertion(task: Any) -> Any:
    criteria = task.evaluation_criteria
    basis = [
        item
        for item in (criteria.reward_basis or [])
        if getattr(item, "value", item) != "NL_ASSERTION"
    ]
    return task.model_copy(
        update={
            "evaluation_criteria": criteria.model_copy(
                update={"reward_basis": basis, "nl_assertions": None}
            )
        }
    )


def retail_task_ids() -> list[str]:
    """First 30 tasks of the zh retail language binding, not English ids 0-29."""
    install_retail_db_only_tasks()
    from tau2.multilingual.runtime import load_language

    loaded = load_language("zh", "retail", mode="text")
    ids = [task.id for task in loaded.tasks][:RETAIL_N]
    if len(ids) != RETAIL_N:
        raise RuntimeError(f"zh retail binding has {len(loaded.tasks)} tasks, need {RETAIL_N}")
    if len(set(ids)) != RETAIL_N:
        raise RuntimeError(f"the first {RETAIL_N} zh retail task ids are not unique")
    return ids


def install_retail_db_only_tasks() -> None:
    """Drop NL assertions from retail tasks loaded in this process. The corpus file stays.

    The scenario object is left unchanged so its published artifact digest stays stable.
    """
    from tau2.registry import registry
    import tau2.multilingual.runtime as runtime

    current = registry.get_tasks_loader("retail")
    if not getattr(current, "_knowdesk_db_only", False):

        def loader(*args, **kwargs):
            return [_without_nl_assertion(task) for task in current(*args, **kwargs)]

        loader._knowdesk_db_only = True  # type: ignore[attr-defined]
        registry._tasks["retail"] = loader

    if getattr(runtime.pin_language_artifacts, "_knowdesk_db_only", False):
        return
    original = runtime.pin_language_artifacts

    def pin_language_artifacts(config, *, update: bool = True):
        loaded = original(config, update=update)
        if getattr(config, "language", None) != "zh" or getattr(config, "domain", None) != "retail":
            return loaded
        return loaded.model_copy(
            update={"tasks": [_without_nl_assertion(task) for task in loaded.tasks]}
        )

    pin_language_artifacts._knowdesk_db_only = True  # type: ignore[attr-defined]
    runtime.pin_language_artifacts = pin_language_artifacts


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


def _replay_skipped_db(path: Path) -> int:
    """Grade DB Match for traces τ left unscored after a premature stop.

    τ writes reward 0 and an empty db_check when termination is not agent_stop
    or user_stop. This replays the saved tool calls with EnvironmentEvaluator
    and stores that hash comparison. It does not call a model or change reward.
    """
    from tau2.data_model.simulation import Results
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.multilingual.runtime import language_environment_kwargs
    from tau2.registry import registry
    from tau2.scripts.evaluate_trajectories import _recorded_language

    raw = json.loads(path.read_text(encoding="utf-8"))
    pending = [
        item
        for item in raw.get("simulations") or []
        if ((item.get("reward_info") or {}).get("db_check") or {}).get("db_reward") is None
    ]
    if not pending:
        return 0
    results = Results.model_validate(raw)
    loaded = _recorded_language(results)
    if loaded is None:
        raise RuntimeError(f"{path} has unscored DB checks and no recorded language binding")
    env_kwargs = language_environment_kwargs(loaded)
    tasks = {task.id: task for task in results.tasks}
    constructor = registry.get_env_constructor("retail")
    scored: dict[str, float] = {}
    for simulation in results.simulations:
        info = simulation.reward_info
        if info is not None and info.db_check is not None and info.db_check.db_reward is not None:
            continue
        task = tasks.get(simulation.task_id)
        if task is None:
            raise RuntimeError(f"{path} is missing task {simulation.task_id}")
        replay = EnvironmentEvaluator.calculate_reward(
            environment_constructor=constructor,
            task=task,
            full_trajectory=list(simulation.messages),
            solo_mode=False,
            env_kwargs=env_kwargs,
            strict_replay=True,
        )
        if replay.db_check is None or replay.db_check.db_reward is None:
            raise RuntimeError(f"{simulation.task_id} replay produced no DB reward")
        scored[simulation.task_id] = float(replay.db_check.db_reward)
    for item in raw.get("simulations") or []:
        task_id = str(item.get("task_id"))
        if task_id not in scored:
            continue
        info = item.setdefault("reward_info", {})
        info["db_check"] = {"db_match": scored[task_id] == 1.0, "db_reward": scored[task_id]}
        breakdown = info.get("reward_breakdown")
        if not isinstance(breakdown, dict):
            info["reward_breakdown"] = {"DB": scored[task_id]}
        else:
            breakdown["DB"] = scored[task_id]
        note = info.get("info")
        if not isinstance(note, dict):
            note = {}
            info["info"] = note
        note["db_replay"] = "tau2 EnvironmentEvaluator on the saved tool trace"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(scored)


def _summarize(path: Path, task_ids: list[str]) -> dict[str, Any]:
    _replay_skipped_db(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(item.get("task_id")): item for item in data.get("simulations") or []}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    rewards = []
    db_scores = []
    nl_scores = []
    assistant_steps: list[int] = []
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
        messages = simulation.get("messages") or []
        assistant_steps.append(sum(1 for message in messages if message.get("role") == "assistant"))
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
        "avg_steps": (sum(assistant_steps) / len(assistant_steps)) if assistant_steps else None,
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
    install_retail_db_only_tasks()
    task_ids = retail_task_ids()
    cells = (
        {
            "key": "tau_retail",
            "label": "llm_agent",
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
            protocol=DB_ONLY_PROTOCOL,
        )
        try:
            run = run_tau(
                exp="x9",
                seed=SEED,
                extra_argv=["--communicate-scorer", "substring"],
                **kwargs,
            )
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
        f"# X9 中文 retail 前 {RETAIL_N} 单（DB Match）",
        "",
        f"任务号是中文 retail 语言包的前 {RETAIL_N} 个（绑定里共 50 条，id 形如 `6_zh_identity`）。",
        f"max_steps={RETAIL_MAX_STEPS}，语言 `zh`，检索 `builtin`。",
        "本表只报告确定性 DB Match。进程内从奖励依据去掉了 `NL_ASSERTION`，communicate 使用 substring，不调用 OpenAI NL 裁判。",
        "τ 在 max_steps 提前结束时不写 db_check。这些轨迹用 EnvironmentEvaluator 重放已保存的工具调用，按数据库哈希补上 DB Match，不改官方 reward，也不再调用模型。",
        "平均步数是轨迹里 assistant 消息条数的均值。重放失败的任务不补 0。",
        "",
        "| 配置 | DB Match | DB 任务数 | 平均步数 | Infra Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {_fmt(summary['db_match'])} | {summary['db_n']} | "
            f"{_fmt(summary['avg_steps'])} | {summary['infra_errors']} |"
        )
    lines.extend(["", "## 对照", ""])
    if baseline["db_match"] is None or candidate["db_match"] is None:
        lines.append("有配置没有 DB Match，不能比较。")
    else:
        lines.append(
            f"knowdesk_graph 的 DB Match `{candidate['db_match']:.4f}` "
            f"{_relation(candidate['db_match'], baseline['db_match'])} "
            f"llm_agent `{baseline['db_match']:.4f}`。"
        )
        lines.append(
            f"平均步数 knowdesk_graph `{candidate['avg_steps']:.2f}`，"
            f"llm_agent `{baseline['avg_steps']:.2f}`。"
        )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    write_manifest(
        "x9",
        model=DEFAULT_MODEL,
        llm_args=dict(DEFAULT_LLM_ARGS),
        seed=SEED,
        split_name=f"retail_zh_first{RETAIL_N}",
        task_ids=task_ids,
        max_steps=RETAIL_MAX_STEPS,
        command="uv run --no-sync python eval/run_x9_retail_zh.py",
        metrics={
            summary["key"]: {key: value for key, value in summary.items() if key != "buckets"}
            | {"bucket_counts": {name: len(ids) for name, ids in summary["buckets"].items()}}
            for summary in summaries
        },
        extra={
            "nl_judge": "stripped",
            "communicate_scorer": "substring",
            "retrieval": "builtin",
            "language": "zh",
            "protocol": DB_ONLY_PROTOCOL,
        },
    )
    infra = sum(summary["infra_errors"] for summary in summaries)
    missing = sum(len(summary["buckets"]["missing_reward"]) for summary in summaries)
    missing_tasks = sum(len(summary["buckets"]["missing_task"]) for summary in summaries)
    db_short = sum(summary["db_n"] != RETAIL_N for summary in summaries)
    print(f"wrote {REPORT_PATH}")
    if infra or missing or missing_tasks or db_short or any(summary["scored"] != RETAIL_N for summary in summaries):
        raise RuntimeError(
            f"retail run is not clean: infra={infra}, missing_reward={missing}, "
            f"missing_task={missing_tasks}, db_short={db_short}. The report was written before this error."
        )


if __name__ == "__main__":
    main()
