"""X1 retrieval benchmark.

Hyperparameters are chosen only on the dev colloquial queries. The test split
is scored afterwards with those frozen values. Hybrid and rerank latencies are
the full query path; this script does not time a cached fusion call.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gate import load_splits, report_ids, require_tunable
from eval.manifest import write_manifest
from eval.x1_select import INTENT_BETAS, select_beta, select_fusion
from knowdesk.kb.eval_retrieval import EvalQuery, load_eval_queries, recall_at_k, reciprocal_rank
from knowdesk.kb.hybrid_retriever import KnowDeskHybridRetriever
from knowdesk.kb.loader import load_documents
from knowdesk.kb.retrieval_config import RetrievalDefaults, save_retrieval_defaults
from knowdesk.kb.router import evaluate_multilabel_f1, map_doc_to_product_line, predict_product_lines_keyword

ROOT = Path(__file__).resolve().parents[1]
COLLOQUIAL_PATH = ROOT / "kb" / "colloquial_queries.json"
REPORT_PATH = ROOT / "docs/benchmarks/RESULTS_X1.md"
SEED = 42


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise RuntimeError("no latency samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_colloquial() -> dict[str, dict]:
    if not COLLOQUIAL_PATH.exists():
        raise FileNotFoundError(COLLOQUIAL_PATH)
    raw = json.loads(COLLOQUIAL_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{COLLOQUIAL_PATH} does not contain queries")
    return {str(item["task_id"]): item for item in raw}


def build_queries(
    task_ids: list[str],
    by_id: dict[str, EvalQuery],
    colloquial: dict[str, dict],
    kind: str,
) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for task_id in task_ids:
        base = by_id.get(task_id)
        if base is None:
            raise RuntimeError(f"{task_id} has no gold retrieval query")
        if kind == "original":
            text = base.query
        elif kind == "colloquial":
            item = colloquial.get(task_id)
            if item is None or not str(item.get("colloquial_query", "")).strip():
                raise RuntimeError(f"{task_id} has no colloquial query")
            text = str(item["colloquial_query"])
        else:
            raise RuntimeError(kind)
        queries.append(EvalQuery(task_id=task_id, query=text, gold_doc_ids=base.gold_doc_ids))
    return queries


def score_lists(
    queries: list[EvalQuery],
    ranked: list[list[str]],
    latencies_ms: list[float],
    *,
    full_path: bool,
) -> dict[str, float]:
    if not queries or len(ranked) != len(queries) or len(latencies_ms) != len(queries):
        raise RuntimeError("retrieval measurement is incomplete")
    count = len(queries)

    def mean(function) -> float:
        return sum(function(docs, query.gold_doc_ids) for docs, query in zip(ranked, queries)) / count

    metrics = {
        "Recall@5": mean(lambda docs, gold: recall_at_k(docs, gold, 5)),
        "Recall@10": mean(lambda docs, gold: recall_at_k(docs, gold, 10)),
        "Recall@20": mean(lambda docs, gold: recall_at_k(docs, gold, 20)),
        "MRR": mean(reciprocal_rank),
        "P50_ms": percentile(latencies_ms, 0.50),
        "P90_ms": percentile(latencies_ms, 0.90),
        "n": count,
    }
    if full_path and metrics["P50_ms"] <= 0.0:
        raise RuntimeError(f"end-to-end P50 collapsed to {metrics['P50_ms']} ms")
    return metrics


def measure(name: str, queries: list[EvalQuery], retrieve, *, family: str, full_path: bool, **meta) -> dict:
    ranked: list[list[str]] = []
    latencies: list[float] = []
    for index, query in enumerate(queries, start=1):
        doc_ids, elapsed_ms = retrieve(query.query)
        if not isinstance(doc_ids, list):
            raise RuntimeError(f"{name} did not return a ranked list")
        ranked.append([str(doc_id) for doc_id in doc_ids])
        latencies.append(float(elapsed_ms))
        if index == len(queries) or index % 25 == 0:
            print(f"  {name}: {index}/{len(queries)}", flush=True)
    row = {
        "name": name,
        "family": family,
        "full_path": full_path,
        "ranked": ranked,
        **score_lists(queries, ranked, latencies, full_path=full_path),
        **meta,
    }
    print(
        f"  {name}: R@10={row['Recall@10']:.4f} MRR={row['MRR']:.4f} "
        f"P50={row['P50_ms']:.1f}ms P90={row['P90_ms']:.1f}ms",
        flush=True,
    )
    return row


def markdown_table(title: str, rows: list[dict]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| 检索配置 | Recall@5 | Recall@10 | Recall@20 | MRR | P50 ms | P90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['Recall@5']:.4f} | {row['Recall@10']:.4f} | "
            f"{row['Recall@20']:.4f} | {row['MRR']:.4f} | {row['P50_ms']:.1f} | {row['P90_ms']:.1f} |"
        )
    lines.append("")
    return lines


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "ranked"}


def same_ranking(left: list[list[str]], right: list[list[str]]) -> bool:
    return left == right


def main() -> None:
    splits = load_splits()
    dev_ids = report_ids(splits, "dev")
    test_ids = report_ids(splits, "test")
    require_tunable(dev_ids, splits)
    if len(dev_ids) != 30 or len(test_ids) != 67:
        raise RuntimeError(f"expected dev 30 and test 67, got {len(dev_ids)} and {len(test_ids)}")
    if set(dev_ids) & set(test_ids):
        raise RuntimeError("dev and test overlap")

    colloquial = load_colloquial()
    task_queries = {query.task_id: query for query in load_eval_queries()}
    dev_colloquial = build_queries(dev_ids, task_queries, colloquial, "colloquial")
    test_colloquial = build_queries(test_ids, task_queries, colloquial, "colloquial")
    dev_original = build_queries(dev_ids, task_queries, colloquial, "original")
    test_original = build_queries(test_ids, task_queries, colloquial, "original")

    mismatches = []
    for task_id, query in task_queries.items():
        item = colloquial.get(task_id)
        if item is None:
            continue
        file_gold = set(item.get("gold_doc_ids") or [])
        if file_gold != query.gold_doc_ids:
            mismatches.append(task_id)
    if mismatches:
        raise RuntimeError(f"colloquial gold differs from task required_documents: {mismatches[:8]}")

    documents = load_documents()
    doc_to_line = {item.doc_id: map_doc_to_product_line(item.doc_id, item.title) for item in documents}
    doc_rep = {item.doc_id: f"{item.title}\n{item.content[:800]}" for item in documents}

    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.domains.banking_knowledge.retrieval import create_bm25_retrieval_pipeline
    from tau2.knowledge.retrievers.bm25_retriever import BM25Retriever

    pipeline = create_bm25_retrieval_pipeline(get_knowledge_base(), top_k=50)
    if not pipeline.retrievers or not isinstance(pipeline.retrievers[0], BM25Retriever):
        found = type(pipeline.retrievers[0]).__name__ if pipeline.retrievers else "none"
        raise RuntimeError(f"τ pipeline did not instantiate BM25Retriever (found {found})")
    tau_ids = set(pipeline.state.get("bm25_doc_ids") or [])
    local_ids = {item.doc_id for item in documents}
    if tau_ids != local_ids:
        raise RuntimeError(
            f"τ BM25 doc ids do not match the local corpus "
            f"(tau={len(tau_ids)} local={len(local_ids)} overlap={len(tau_ids & local_ids)})"
        )

    print("loading hybrid retriever", flush=True)
    retriever = KnowDeskHybridRetriever.get_instance()

    def timed_tau(query: str) -> tuple[list[str], float]:
        started = time.perf_counter()
        hits = pipeline.retrieve(query, top_k=50)
        return [doc_id for doc_id, _score in hits], (time.perf_counter() - started) * 1000.0

    def timed_rank(query: str, **overrides) -> tuple[list[str], float]:
        started = time.perf_counter()
        ranked = retriever.rank_documents(query, k=20, **overrides)
        return [doc_id for doc_id, _score, _chunk in ranked], (time.perf_counter() - started) * 1000.0

    def dev_fusion_rows() -> list[dict]:
        rows = [
            measure(
                "τ 原生 BM25Retriever",
                dev_colloquial,
                timed_tau,
                family="tau_bm25",
                full_path=True,
            )
        ]
        rows.append(measure(
            "静态 RRF（BM25 块 + Qwen 向量，同条件）",
            dev_colloquial,
            lambda query: timed_rank(query, fusion_mode="rrf", use_intent_boost=False, beta=0.0),
            family="rrf",
            full_path=True,
            temperature=None,
            alpha=None,
            beta=0.0,
        ))
        for temperature in (0.5, 1.0, 2.0):
            rows.append(measure(
                f"AdaRRF temperature={temperature:.1f}（同条件，无意图加权）",
                dev_colloquial,
                lambda query, temperature=temperature: timed_rank(
                    query, fusion_mode="adaptive_rrf", temperature=temperature, use_intent_boost=False, beta=0.0
                ),
                family="adarrf",
                full_path=True,
                temperature=temperature,
                alpha=None,
                beta=0.0,
            ))
        for alpha in (0.3, 0.5, 0.7):
            rows.append(measure(
                f"线性融合 alpha={alpha:.1f}（同条件，无意图加权）",
                dev_colloquial,
                lambda query, alpha=alpha: timed_rank(
                    query, fusion_mode="linear", alpha=alpha, use_intent_boost=False, beta=0.0
                ),
                family="linear",
                full_path=True,
                temperature=None,
                alpha=alpha,
                beta=0.0,
            ))
        return rows

    print("dev colloquial fusion sweep", flush=True)
    fusion_rows = dev_fusion_rows()
    fusion_choice = select_fusion(fusion_rows)
    winner = fusion_choice["fusion_row"]
    print(f"dev fusion winner: {winner['name']}", flush=True)

    beta_rows = []
    for beta in INTENT_BETAS:
        if beta == 0.0:
            continue
        overrides = {
            "fusion_mode": {"adarrf": "adaptive_rrf", "rrf": "rrf", "linear": "linear"}[winner["family"]],
            "use_intent_boost": True,
            "beta": beta,
        }
        if winner["family"] == "adarrf":
            overrides["temperature"] = winner["temperature"]
        if winner["family"] == "linear":
            overrides["alpha"] = winner["alpha"]
        beta_rows.append(measure(
            f"{winner['name']} + intent beta={beta:.2f}",
            dev_colloquial,
            lambda query, overrides=overrides: timed_rank(query, **overrides),
            family="beta",
            full_path=True,
            beta=beta,
            temperature=winner.get("temperature"),
            alpha=winner.get("alpha"),
        ))
    frozen = select_beta(winner, beta_rows)
    saved = RetrievalDefaults(
        fusion_mode=frozen["fusion_mode"],
        temperature=float(frozen["temperature"]),
        alpha=float(frozen["alpha"]),
        beta=float(frozen["beta"]),
        use_intent_boost=bool(frozen["use_intent_boost"]),
        dense_model=retriever.dense.model_name,
        selected_on="dev_colloquial",
    )
    save_retrieval_defaults(saved)
    print(f"froze online defaults: {saved.to_dict()}", flush=True)

    def timed_default(query: str) -> tuple[list[str], float]:
        started = time.perf_counter()
        ranked = retriever.rank_documents(query, k=20)
        return [doc_id for doc_id, _score, _chunk in ranked], (time.perf_counter() - started) * 1000.0

    online_dev = measure(
        "在线默认（dev 冻结后原样调用 rank_documents）",
        dev_colloquial,
        timed_default,
        family="online",
        full_path=True,
    )
    if not same_ranking(online_dev["ranked"], frozen["selected_row"]["ranked"]):
        raise RuntimeError("online default ranking does not match the dev-selected row")

    from knowdesk.kb.rerank import LocalReranker

    reranker = LocalReranker("BAAI/bge-reranker-v2-m3")

    def timed_rerank(query: str) -> tuple[list[str], float]:
        started = time.perf_counter()
        base = retriever.rank_documents(query, k=30)
        candidates = [(doc_id, doc_rep[doc_id]) for doc_id, _score, _chunk in base if doc_id in doc_rep]
        scores = reranker.score_pairs(query, [text for _doc_id, text in candidates])
        ordered = [doc_id for doc_id, _score in sorted(zip((doc_id for doc_id, _text in candidates), scores), key=lambda item: item[1], reverse=True)]
        return ordered[:20], (time.perf_counter() - started) * 1000.0

    rerank_dev = measure(
        "在线默认 Top-30 + bge-reranker-v2-m3（消融，不进入默认）",
        dev_colloquial,
        timed_rerank,
        family="rerank",
        full_path=True,
    )

    def frozen_overrides() -> dict:
        overrides: dict = {"fusion_mode": saved.fusion_mode, "use_intent_boost": saved.use_intent_boost, "beta": saved.beta}
        if saved.fusion_mode == "adaptive_rrf":
            overrides["temperature"] = saved.temperature
        if saved.fusion_mode == "linear":
            overrides["alpha"] = saved.alpha
        return overrides

    explicit = frozen_overrides()

    ada_temperature = float(fusion_choice["best_adarrf"]["temperature"])
    linear_alpha = float(fusion_choice["best_linear"]["alpha"])

    def measure_frozen(queries: list[EvalQuery], *, include_rerank: bool) -> list[dict]:
        rows = [
            measure("τ 原生 BM25Retriever", queries, timed_tau, family="tau_bm25", full_path=True),
            measure(
                "静态 RRF（同条件）",
                queries,
                lambda query: timed_rank(query, fusion_mode="rrf", use_intent_boost=False, beta=0.0),
                family="rrf",
                full_path=True,
            ),
            measure(
                f"线性融合 alpha={linear_alpha:.1f}（dev 上该族最优，同条件）",
                queries,
                lambda query: timed_rank(
                    query, fusion_mode="linear", alpha=linear_alpha, use_intent_boost=False, beta=0.0
                ),
                family="linear",
                full_path=True,
                alpha=linear_alpha,
            ),
            measure(
                f"AdaRRF temperature={ada_temperature:.1f}（dev 上该族最优，同条件）",
                queries,
                lambda query: timed_rank(
                    query,
                    fusion_mode="adaptive_rrf",
                    temperature=ada_temperature,
                    use_intent_boost=False,
                    beta=0.0,
                ),
                family="adarrf",
                full_path=True,
                temperature=ada_temperature,
            ),
            measure(
                "在线默认（dev 冻结配置）",
                queries,
                lambda query: timed_rank(query, **explicit),
                family="online",
                full_path=True,
            ),
        ]
        default_row = measure("在线默认（无覆盖参数）", queries, timed_default, family="online_default_call", full_path=True)
        if not same_ranking(default_row["ranked"], rows[-1]["ranked"]):
            raise RuntimeError("frozen config and unscoped rank_documents disagree")
        rows[-1] = default_row
        rows[-1]["name"] = "在线默认（dev 冻结配置，无覆盖参数）"
        if include_rerank:
            rows.append(measure(
                "在线默认 Top-30 + bge-reranker-v2-m3（消融，不进入默认）",
                queries,
                timed_rerank,
                family="rerank",
                full_path=True,
            ))
        return rows

    print("test colloquial frozen report", flush=True)
    test_rows = measure_frozen(test_colloquial, include_rerank=True)
    print("original-instruction frozen report (not used for selection)", flush=True)
    dev_original_rows = measure_frozen(dev_original, include_rerank=False)
    test_original_rows = measure_frozen(test_original, include_rerank=False)

    dev_table_rows = [row for row in fusion_rows if row["family"] != "tau_bm25"]
    # Keep the tau baseline and the selected beta rows in the dev table.
    dev_report_rows = fusion_rows + beta_rows + [online_dev, rerank_dev]

    def intent_f1(queries: list[EvalQuery]) -> dict[str, float]:
        y_true = [{doc_to_line[doc_id] for doc_id in query.gold_doc_ids if doc_id in doc_to_line} for query in queries]
        y_pred = [predict_product_lines_keyword(query.query) for query in queries]
        return evaluate_multilabel_f1(y_true, y_pred)

    dev_f1 = intent_f1(dev_colloquial)
    test_f1 = intent_f1(test_colloquial)
    lines = [
        "# X1 检索实验（dev 调参 / test 冻结报告）",
        "",
        f"- 随机种子 `{SEED}`。超参只在 dev {len(dev_ids)} 条口语查询上选择，选择指标是 Recall@10，平手再看 MRR，再看更低的 P50。",
        f"- test {len(test_ids)} 只用于下面的冻结报告，没有参与 `temperature`、`alpha` 或 `beta` 的选择。",
        f"- 在线默认配置：`{json.dumps(saved.to_dict(), ensure_ascii=False)}`。",
        f"- 第 1 行调用 τ `BM25Retriever`。混合行与重排行的延迟包含当次 BM25、查询向量编码、融合，以及重排行自己的重排。",
        f"- 重排是消融行，不因为 test 或 dev 上的 Recall 更高就写进在线默认。",
        f"- dev 意图路由 Micro-F1 `{dev_f1['micro_f1']:.4f}`，Macro-F1 `{dev_f1['macro_f1']:.4f}`。",
        f"- test 意图路由 Micro-F1 `{test_f1['micro_f1']:.4f}`，Macro-F1 `{test_f1['macro_f1']:.4f}`。这项不用于选择检索配置。",
        "",
    ]
    lines.extend(markdown_table(f"表 1：dev 口语查询调参（n={len(dev_colloquial)}）", dev_report_rows))
    lines.extend(markdown_table(f"表 2：test 口语查询冻结报告（n={len(test_colloquial)}）", test_rows))
    lines.extend(markdown_table(
        f"附录 A：dev 原始工单说明，使用冻结配置，不参与选择（n={len(dev_original)}）",
        dev_original_rows,
    ))
    lines.extend(markdown_table(
        f"附录 B：test 原始工单说明，使用冻结配置，不参与选择（n={len(test_original)}）",
        test_original_rows,
    ))
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")

    def pack(rows: list[dict]) -> list[dict]:
        return [public_row(row) for row in rows]

    write_manifest(
        "x1",
        model=retriever.dense.model_name,
        llm_args={"reranker": "BAAI/bge-reranker-v2-m3", "retrieval": saved.to_dict()},
        seed=SEED,
        split_name="dev30_tune_test67_report",
        task_ids={"dev": dev_ids, "test": test_ids},
        max_steps=None,
        command="uv run --no-sync python eval/run_x1_full.py",
        metrics={
            "dev_colloquial": pack(dev_report_rows),
            "test_colloquial": pack(test_rows),
            "dev_original_frozen": pack(dev_original_rows),
            "test_original_frozen": pack(test_original_rows),
            "intent_f1": {"dev_colloquial": dev_f1, "test_colloquial": test_f1},
        },
        extra={
            "tuned_on": "dev_colloquial",
            "selected_config": saved.to_dict(),
            "selection_rule": "max Recall@10, then MRR, then lower P50; rerank excluded",
            "tau_retriever": "tau2.knowledge.retrievers.bm25_retriever.BM25Retriever",
            "unused_dev_rows_for_selection": [public_row(row) for row in dev_table_rows],
        },
    )
    print(report)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
