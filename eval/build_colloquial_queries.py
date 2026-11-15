import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
import litellm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb.loader import load_documents, DEFAULT_TASKS_PATH
from kb.eval_retrieval import load_eval_queries, EvalQuery

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "kb/colloquial_queries.json"
PROMPT_VERSION = "m4_colloquial_v1"
MODEL_NAME = "dashscope/qwen3.8-flash"

SYSTEM_PROMPT = (
    "You are simulating a real bank customer contacting customer support. "
    "Rewrite the given scenario description into the customer's VERY FIRST spoken turn (1 to 2 natural, colloquial English sentences). "
    "Rules:\n"
    "1. Do NOT use internal policy document titles or internal tool names.\n"
    "2. Sound like an everyday customer explaining their problem or question naturally.\n"
    "3. Output ONLY the customer's first turn text, nothing else."
)

def rewrite_one(eq: EvalQuery) -> dict:
    resp = litellm.completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scenario:\n{eq.query}"},
        ],
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    colloquial = resp.choices[0].message.content.strip().strip('"')
    return {
        "task_id": eq.task_id,
        "original_query": eq.query,
        "colloquial_query": colloquial,
        "gold_doc_ids": sorted(list(eq.gold_doc_ids)),
        "model": getattr(resp, "model", MODEL_NAME),
        "prompt_version": PROMPT_VERSION,
    }

def main():
    env_path = Path(__file__).resolve().parents[2] / "tau2-bench/.env"
    load_dotenv(env_path, override=True)

    queries = load_eval_queries(DEFAULT_TASKS_PATH)
    docs = {d.doc_id: d.title for d in load_documents()}

    print(f"🔄 正在用 {MODEL_NAME} 并发生成 {len(queries)} 条口语化首轮查询...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(rewrite_one, queries))

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 已保存至 {OUTPUT_PATH}")

    leak_count = 0
    for item in results:
        q_low = item["colloquial_query"].lower()
        gold_titles = [docs[did].lower() for did in item["gold_doc_ids"] if did in docs]
        if any(t in q_low for t in gold_titles if len(t) > 8):
            leak_count += 1

    print(f"🔎 标题原样泄漏检查: {leak_count}/{len(results)} ({leak_count/len(results)*100:.2f}%)")
    print("\n📋 随机抽检前 3 条：")
    for item in results[:3]:
        print(f"  [{item['task_id']}] {item['colloquial_query']}")

if __name__ == "__main__":
    main()
