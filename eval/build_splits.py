import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb.loader import DEFAULT_TASKS_PATH, load_documents
from kb.router import map_doc_to_product_line

SPLITS_PATH = Path(__file__).resolve().parent / "splits.json"
SEED = 42

def main():
    raw_tasks = json.loads(DEFAULT_TASKS_PATH.read_text(encoding="utf-8"))
    docs = {d.doc_id: map_doc_to_product_line(d.doc_id, d.title) for d in load_documents()}

    # 按 (是否跨产品大类, 是否短链路 gold_actions <= 5) 分为 4 个桶做分层抽样
    buckets: dict[tuple[bool, bool], list[dict]] = {}
    for t in raw_tasks:
        tid = str(t["id"])
        req_docs = t.get("required_documents", [])
        lines = {docs.get(str(d), "general_policy") for d in req_docs}
        is_cross_line = len(lines) > 1
        actions = t.get("evaluation_criteria", {}).get("actions", [])
        is_short = len(actions) <= 5
        meta = {
            "task_id": tid,
            "is_cross_line": is_cross_line,
            "is_short": is_short,
            "num_gold_actions": len(actions),
            "num_req_docs": len(req_docs),
        }
        buckets.setdefault((is_cross_line, is_short), []).append(meta)

    rng = random.Random(SEED)
    dev_meta: list[dict] = []
    test_meta: list[dict] = []

    total = len(raw_tasks)
    target_dev = 30
    for key in sorted(buckets.keys()):
        items = sorted(buckets[key], key=lambda x: x["task_id"])
        rng.shuffle(items)
        # 按比例分配 dev 数量
        k_dev = round(len(items) * target_dev / total)
        dev_meta.extend(items[:k_dev])
        test_meta.extend(items[k_dev:])

    # 微调确保恰好 dev=30, test=67
    while len(dev_meta) > target_dev:
        test_meta.append(dev_meta.pop())
    while len(dev_meta) < target_dev:
        dev_meta.append(test_meta.pop())

    dev_ids = sorted(x["task_id"] for x in dev_meta)
    test_ids = sorted(x["task_id"] for x in test_meta)
    short_dev_ids = sorted(x["task_id"] for x in dev_meta if x["is_short"])
    long_dev_ids = sorted(x["task_id"] for x in dev_meta if not x["is_short"])
    # M19 回归门禁集：6个短链路 + 6个长链路 = 12个任务
    regression_ids = sorted(short_dev_ids[:6] + long_dev_ids[:6])

    payload = {
        "seed": SEED,
        "total_tasks": total,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "regression_ids": regression_ids,
        "short_dev_ids": short_dev_ids,
        "stats": {
            "dev_count": len(dev_ids),
            "test_count": len(test_ids),
            "regression_count": len(regression_ids),
            "dev_short_ratio": round(sum(1 for x in dev_meta if x["is_short"]) / len(dev_meta), 4),
            "test_short_ratio": round(sum(1 for x in test_meta if x["is_short"]) / len(test_meta), 4),
            "dev_cross_line_ratio": round(sum(1 for x in dev_meta if x["is_cross_line"]) / len(dev_meta), 4),
            "test_cross_line_ratio": round(sum(1 for x in test_meta if x["is_cross_line"]) / len(test_meta), 4),
        },
    }
    SPLITS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ 分层切分已保存至 eval/splits.json:")
    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
