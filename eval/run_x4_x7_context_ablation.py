"""X5 online compaction and X7 retrieved-context measurement.

--x7-only measures the frozen retriever on all 97 colloquial queries. It does
not call Bailian and it does not multiply a corpus-wide average.
--online runs compaction off and on for knowdesk_graph + hybrid on dev 30.
Without one of those flags the script raises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gate import load_splits, report_ids
from eval.manifest import write_manifest
from eval.metrics import rewards_for_tasks, summarize_simulation_file
from eval.pass_k import paired_bootstrap_diff
from eval.tau_runner import DEFAULT_LLM_ARGS, DEFAULT_MODEL, run_tau
from eval.x2_protocol import GRAPH_HYBRID_KEY, PROTOCOL_MAX_STEPS, SEED, cell_by_key

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "RESULTS_X4_X7_ABLATION.md"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
X5_METRICS = RESULTS_DIR / "x5_metrics.json"
X7_METRICS = RESULTS_DIR / "x7_metrics.json"
X5_SELECTED = RESULTS_DIR / "x5_selected.json"


def summarize_retrieval_chars(per_query: list[dict[str, Any]]) -> dict[str, float | int]:
    """Average the characters of hits that were actually returned."""
    if not per_query:
        raise RuntimeError("no queries to measure")
    full_hits = [chars for query in per_query for chars in query["full_chars"]]
    section_hits = [chars for query in per_query for chars in query["section_chars"]]
    faithful_flags = [flag for query in per_query for flag in query["section_faithful"]]
    if not full_hits or not section_hits or len(section_hits) != len(faithful_flags):
        raise RuntimeError("retrieval character measurement is incomplete")
    query_full = [sum(query["full_chars"]) for query in per_query]
    query_section = [sum(query["section_chars"]) for query in per_query]
    return {
        "n_queries": len(per_query),
        "n_full_hits": len(full_hits),
        "n_section_hits": len(section_hits),
        "queries_short_of_10_docs": sum(len(query["full_chars"]) < 10 for query in per_query),
        "queries_short_of_5_sections": sum(len(query["section_chars"]) < 5 for query in per_query),
        "mean_full_doc_chars": sum(full_hits) / len(full_hits),
        "mean_section_body_chars": sum(section_hits) / len(section_hits),
        "mean_query_full_chars": sum(query_full) / len(query_full),
        "mean_query_section_chars": sum(query_section) / len(query_section),
        "clause_fidelity": sum(1 for flag in faithful_flags if flag) / len(faithful_flags),
    }


def render_report() -> None:
    x5 = json.loads(X5_METRICS.read_text(encoding="utf-8")) if X5_METRICS.exists() else None
    x7 = json.loads(X7_METRICS.read_text(encoding="utf-8")) if X7_METRICS.exists() else None
    lines = [
        "# X5 在线压缩与 X7 检索上下文字符",
        "",
        "没有对应测量文件的表不填通过率或字符数。",
        "",
        "## X5 dev 30 在线压缩",
        "",
    ]
    if x5 is None:
        lines.append("X5 尚未在线运行。这里不填写 Pass^1。")
    else:
        lines.extend([
            f"配置固定为 `{x5['config']}`，max_steps `{x5['max_steps']}`。压缩开关只改变 `KNOWDESK_COMPACTION`。",
            f"选择结果：`compaction={x5['selected_compaction']}`，判定 `{x5['verdict']}`。",
            "",
            "| 设置 | Pass^1 | 平均输入 token | 平均 cached token | cache_hit_rate | 平均消息条数 | 平均 LLM 调用 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in x5["rows"]:
            cached = "n/a" if row["avg_cached_tokens"] is None else f"{row['avg_cached_tokens']:.0f}"
            hit = "n/a" if row["cache_hit_rate"] is None else f"{row['cache_hit_rate']:.4f}"
            lines.append(
                f"| {row['label']} | {row['pass_1']:.4f} | {row['avg_in_tokens']:.0f} | "
                f"{cached} | {hit} | {row['avg_messages']:.1f} | {row['avg_llm_calls']:.1f} |"
            )
        lines.extend([
            "",
            f"配对差 `{x5['mean_diff']:.4f}`，95% CI `[{x5['ci_95_low']:.4f}, {x5['ci_95_high']:.4f}]`。",
            "压缩开启只在统计上更好，或在非劣且平均输入 token 下降时被选中。否则保持关闭。",
        ])
    lines.extend(["", "## X7 实际命中的上下文字符", ""])
    if x7 is None:
        lines.append("X7 尚未测量。")
    else:
        lines.extend([
            f"冻结配置 `{json.dumps(x7['retrieval'], ensure_ascii=False)}`。",
            f"查询数 `{x7['n_queries']}`。不足 10 篇文档的查询 `{x7['queries_short_of_10_docs']}`，"
            f"不足 5 个章节的查询 `{x7['queries_short_of_5_sections']}`。",
            "",
            "| 指标 | 值 |",
            "|---|---:|",
            f"| 单篇命中文档平均字符 | {x7['mean_full_doc_chars']:.1f} |",
            f"| 单个命中章节 body 平均字符 | {x7['mean_section_body_chars']:.1f} |",
            f"| 每查询 top-10 文档字符和的平均 | {x7['mean_query_full_chars']:.1f} |",
            f"| 每查询 top-5 章节字符和的平均 | {x7['mean_query_section_chars']:.1f} |",
            f"| 章节 body 出现在父文档中的比例 | {x7['clause_fidelity']:.4f} |",
            "",
            "字符数来自这 97 次检索实际返回的文档和章节，不是语料库平均长度相乘。",
        ])
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_x7() -> dict[str, Any]:
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from kb.hybrid_retriever import KnowDeskHybridRetriever
    from kb.retrieval_config import CONFIG_PATH, load_retrieval_defaults

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "kb/retrieval_defaults.json is missing. Run eval/run_x1_full.py before X7."
        )
    defaults = load_retrieval_defaults()
    if defaults.selected_on != "dev_colloquial":
        raise RuntimeError(
            f"retrieval defaults were selected on {defaults.selected_on!r}, not dev_colloquial"
        )
    colloquial_path = ROOT / "kb" / "colloquial_queries.json"
    colloquial = json.loads(colloquial_path.read_text(encoding="utf-8"))
    if len(colloquial) != 97:
        raise RuntimeError(f"expected 97 colloquial queries, got {len(colloquial)}")
    splits = load_splits()
    expected_ids = set(report_ids(splits, "dev")) | set(report_ids(splits, "test"))
    got_ids = {str(item["task_id"]) for item in colloquial}
    if got_ids != expected_ids:
        raise RuntimeError("colloquial query ids are not the dev 30 plus test 67 split")
    retriever = KnowDeskHybridRetriever.get_instance()
    per_query = []
    for index, item in enumerate(colloquial, start=1):
        query = str(item.get("colloquial_query") or "").strip()
        if not query:
            raise RuntimeError(f"{item.get('task_id')} has no colloquial query")
        ranked = retriever.rank_documents(query, k=10)
        full_chars = []
        section_chars = []
        faithful = []
        for doc_id, _score, chunk in ranked:
            document = retriever.doc_map[doc_id]
            full_chars.append(len(document.content))
            if len(section_chars) < 5:
                body = chunk.body or ""
                parent = document.content.replace("\r\n", "\n")
                section_chars.append(len(body))
                faithful.append(bool(body) and body in parent)
        per_query.append({
            "task_id": str(item["task_id"]),
            "full_chars": full_chars,
            "section_chars": section_chars,
            "section_faithful": faithful,
        })
        if index == len(colloquial) or index % 25 == 0:
            print(f"x7 {index}/{len(colloquial)}", flush=True)
    summary = summarize_retrieval_chars(per_query)
    payload = {**summary, "retrieval": defaults.to_dict(), "task_ids": [row["task_id"] for row in per_query]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X7_METRICS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(
        "x7",
        model=defaults.dense_model,
        llm_args=defaults.to_dict(),
        seed=SEED,
        split_name="all_97_colloquial_frozen_retriever",
        task_ids=payload["task_ids"],
        max_steps=None,
        command="uv run --no-sync python eval/run_x4_x7_context_ablation.py --x7-only",
        metrics=summary,
    )
    render_report()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return payload


def _row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "pass_1": metrics["pass_1"],
        "avg_in_tokens": metrics["avg_in_tokens"],
        "avg_cached_tokens": metrics["avg_cached_tokens"],
        "cache_hit_rate": metrics["cache_hit_rate"],
        "avg_messages": metrics["avg_messages"],
        "avg_llm_calls": metrics["avg_llm_calls"],
    }


def run_x5() -> dict[str, Any]:
    splits = load_splits()
    task_ids = report_ids(splits, "dev")
    if len(task_ids) != 30:
        raise RuntimeError(f"X5 runs on dev 30, got {len(task_ids)}")
    base = cell_by_key(GRAPH_HYBRID_KEY)
    measured = []
    for compaction, label in (("0", "compaction off"), ("1", "compaction on")):
        cell = dict(base)
        cell["compaction_mode"] = compaction
        run = run_tau(
            exp="x5",
            domain="banking_knowledge",
            agent=cell["agent"],
            retrieval=cell["retrieval"],
            prompt_version=cell["prompt_version"],
            guard_mode=cell["guard_mode"],
            compaction_mode=compaction,
            task_ids=task_ids,
            max_steps=PROTOCOL_MAX_STEPS,
            num_trials=1,
            seed=SEED,
        )
        metrics = summarize_simulation_file(run["results_path"])
        if metrics["avg_in_tokens"] is None or float(metrics["avg_in_tokens"]) <= 0:
            raise RuntimeError(f"{label} did not report prompt tokens")
        measured.append((compaction, label, run, metrics, rewards_for_tasks(run["results_path"], task_ids)))
    off = measured[0]
    on = measured[1]
    diff = paired_bootstrap_diff(off[4], on[4], seed=SEED)
    tokens_down = float(on[3]["avg_in_tokens"]) < float(off[3]["avg_in_tokens"])
    if diff["verdict"] == "STATISTICALLY_IMPROVED" or (diff["verdict"] == "NON_INFERIOR" and tokens_down):
        selected = "1"
    else:
        selected = "0"
    payload = {
        "config": GRAPH_HYBRID_KEY,
        "max_steps": PROTOCOL_MAX_STEPS,
        "selected_compaction": selected,
        "verdict": diff["verdict"],
        "mean_diff": diff["mean_diff"],
        "ci_95_low": diff["ci_95_low"],
        "ci_95_high": diff["ci_95_high"],
        "rows": [_row(off[1], off[3]), _row(on[1], on[3])],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X5_METRICS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    X5_SELECTED.write_text(json.dumps({
        "compaction_mode": selected,
        "agent": base["agent"],
        "retrieval": base["retrieval"],
        "prompt_version": base["prompt_version"],
        "guard_mode": base["guard_mode"],
        "verdict": diff["verdict"],
        "pass_1_off": off[3]["pass_1"],
        "pass_1_on": on[3]["pass_1"],
        "avg_in_tokens_off": off[3]["avg_in_tokens"],
        "avg_in_tokens_on": on[3]["avg_in_tokens"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(
        "x5",
        model=DEFAULT_MODEL,
        llm_args=dict(DEFAULT_LLM_ARGS),
        seed=SEED,
        split_name="dev30",
        task_ids=task_ids,
        max_steps=PROTOCOL_MAX_STEPS,
        command="uv run --no-sync python eval/run_x4_x7_context_ablation.py --online",
        metrics=payload,
    )
    render_report()
    print(f"selected compaction={selected} verdict={diff['verdict']}")
    return payload


def main() -> None:
    online = "--online" in sys.argv
    x7 = "--x7-only" in sys.argv
    if not online and not x7:
        raise RuntimeError(
            "pass --x7-only for the local character measurement or --online for the paid "
            "dev-30 compaction pair. This script does not invent token totals or pass rates."
        )
    if online:
        run_x5()
    if x7:
        run_x7()


if __name__ == "__main__":
    main()
