"""Sprint 3: eval scripts must fail closed instead of inventing measurements."""

import csv
import json
import sys
from pathlib import Path

import pytest

from eval.injection import evaluate_injection_defense, table_a_records
from eval.manifest import REQUIRED_FIELDS, write_manifest
from eval.metrics import message_usage, rewards_for_tasks
from eval.pass_k import paired_bootstrap_diff
from eval.run_x1_bm25 import main as retired_x1_main
from eval.run_x11_judge_calibration import export_blind_csv, score_blind_csv
from eval.run_x4_x7_context_ablation import main as x4_main, summarize_retrieval_chars
from eval.tau_runner import cache_dirname, require_dashscope_key
from eval.x1_select import select_beta, select_fusion
from kb.retrieval_config import (
    RetrievalDefaults,
    load_retrieval_defaults,
    save_retrieval_defaults,
)

BANNED = (
    "raw_tokens_total = 265590",
    "pass_rate = 0.4000",
    '"human_label": "faithful"',
    "_deterministic_judge",
    "rmap.get(tid, 0.0)",
    "avg_doc_chars * 10",
)


def test_eval_sources_do_not_contain_synthetic_fallbacks():
    root = Path("eval")
    offenders = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for phrase in BANNED:
            if phrase in text:
                offenders.append(f"{path.name}: {phrase}")
    assert offenders == []


def test_pass_k_verdicts_match_the_interval():
    improved = paired_bootstrap_diff([0.0] * 10, [1.0] * 10, seed=42)
    assert improved["verdict"] == "STATISTICALLY_IMPROVED"
    assert improved["non_regressing"] is True

    tied = paired_bootstrap_diff([0.0] * 8, [0.0] * 8, seed=42)
    assert tied["verdict"] == "NON_INFERIOR"
    assert tied["ci_95_low"] == 0.0

    wide = paired_bootstrap_diff([0.0, 1.0] * 5, [1.0, 0.0] * 5, seed=42)
    assert wide["ci_95_low"] < -0.05 < 0.0 < wide["ci_95_high"]
    assert wide["verdict"] == "INCONCLUSIVE"
    assert wide["non_regressing"] is False

    regressed = paired_bootstrap_diff([1.0] * 10, [0.0] * 10, seed=42)
    assert regressed["verdict"] == "REGRESSED"
    assert regressed["non_regressing"] is False

    with pytest.raises(RuntimeError):
        paired_bootstrap_diff([], [])
    with pytest.raises(RuntimeError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_retrieval_defaults_roundtrip(tmp_path: Path):
    path = tmp_path / "retrieval_defaults.json"
    saved = RetrievalDefaults(
        fusion_mode="rrf",
        temperature=1.0,
        alpha=0.5,
        beta=0.0,
        use_intent_boost=False,
        selected_on="dev_colloquial",
    )
    save_retrieval_defaults(saved, path)
    loaded = load_retrieval_defaults(path)
    assert loaded == saved


def test_manifest_contains_required_fields(tmp_path: Path):
    path = write_manifest(
        "unit",
        model="dashscope/qwen3.8-flash",
        llm_args={"temperature": 0},
        seed=42,
        split_name="dev30",
        task_ids=["task_002"],
        max_steps=70,
        command="uv run --no-sync python eval/run_x2.py",
        metrics={"pass_1": 0.0},
        directory=tmp_path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "unit_manifest.json"
    for field in REQUIRED_FIELDS:
        assert field in payload
    assert payload["git_sha"]
    assert isinstance(payload["git_dirty"], bool)


def test_x1_selects_adarrf_temperature_then_beta():
    def row(family: str, recall: float, **extra):
        return {"family": family, "Recall@10": recall, "MRR": 0.4, "P50_ms": 12.0, **extra}

    rows = [
        row("rrf", 0.40),
        row("adarrf", 0.80, temperature=0.5),
        row("adarrf", 0.50, temperature=1.0),
        row("adarrf", 0.45, temperature=2.0),
        row("linear", 0.30, alpha=0.3),
        row("linear", 0.35, alpha=0.5),
        row("linear", 0.33, alpha=0.7),
    ]
    choice = select_fusion(rows)
    assert choice["fusion_row"]["temperature"] == 0.5
    betas = [
        row("adarrf", 0.90, temperature=0.5, beta=0.15),
        row("adarrf", 0.70, temperature=0.5, beta=0.30),
    ]
    selected = select_beta(choice["fusion_row"], betas)
    assert selected["fusion_mode"] == "adaptive_rrf"
    assert selected["temperature"] == 0.5
    assert selected["beta"] == 0.15
    assert selected["use_intent_boost"] is True


def _citation_results(path: Path, count: int) -> None:
    simulations = []
    for index in range(count):
        doc_id = f"doc_policy_case_{index:03d}"
        source = f"{doc_id}\nThe published clause {index} waives the fee when the customer meets the written condition."
        reply = (
            f"The policy {doc_id} says the published clause {index} waives the fee "
            "when the customer meets the written condition."
        )
        simulations.append({
            "task_id": f"task_{index:03d}",
            "messages": [
                {"role": "tool", "name": "KB_search", "content": source},
                {"role": "assistant", "content": reply},
            ],
        })
    path.write_text(json.dumps({"simulations": simulations}), encoding="utf-8")


def test_x11_export_leaves_human_label_blank_and_refuses_to_pad(tmp_path: Path):
    short = tmp_path / "short.json"
    _citation_results(short, 49)
    with pytest.raises(RuntimeError, match="Refusing to pad"):
        export_blind_csv(short, tmp_path / "short.csv", record_manifest=False)

    full = tmp_path / "full.json"
    _citation_results(full, 50)
    export_blind_csv(full, tmp_path / "blind.csv", record_manifest=False)
    with (tmp_path / "blind.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50
    assert all(row["human_label"] == "" for row in rows)
    assert rows[0]["doc_text"].startswith("doc_policy_case_000")


def test_x11_score_raises_before_any_api_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / "blind.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", "task_id", "doc_text", "agent_reply", "human_label"]
        )
        writer.writeheader()
        for index in range(50):
            writer.writerow({
                "item_id": f"x11_{index:03d}",
                "task_id": f"task_{index:03d}",
                "doc_text": "source text " * 8,
                "agent_reply": "reply text " * 8,
                "human_label": "",
            })

    def forbid_api(**_kwargs):
        raise AssertionError("judge API was called")

    monkeypatch.setattr("litellm.completion", forbid_api, raising=False)
    with pytest.raises(RuntimeError, match="human_label"):
        score_blind_csv(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["human_label"] = "faithful"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("eval.tau_runner.load_tau_env", lambda: None)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        score_blind_csv(csv_path)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        require_dashscope_key()


def test_table_a_rates_are_recomputed_from_decisions():
    records = table_a_records()
    reported = evaluate_injection_defense()
    assert len(records) == 30
    payload = [row for row in records if row["kind"] == "payload"]
    unverified = [row for row in records if row["kind"] == "unverified"]
    assert reported["violation_rate_guard_off"] == 1.0
    assert reported["guard_off_definitional"] is True
    assert reported["violation_rate_guard_on_payload"] == (
        sum(row["action"] == "allow" for row in payload) / len(payload)
    )
    assert reported["violation_rate_guard_on_unverified"] == (
        sum(row["action"] == "allow" for row in unverified) / len(unverified)
    )
    assert all(row["action"] == "block" and row["rule_id"] == "R02_PROMPT_INJECTION_PAYLOAD" for row in payload)
    assert all(row["action"] == "block" and row["rule_id"] == "R03_REQUIRE_UNLOCK_BEFORE_DISCOVERABLE_CALL" for row in unverified)


def test_message_usage_reads_cached_tokens_from_raw_data():
    usage = message_usage({
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        "raw_data": {"usage": {"prompt_tokens_details": {"cached_tokens": 40}}},
    })
    assert usage["cached_tokens"] == 40
    assert "cached_tokens" not in message_usage({"usage": {"prompt_tokens": 10}})


def test_missing_task_reward_raises(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "simulations": [{"task_id": "task_002", "reward_info": {"reward": 1.0}}]
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing tasks"):
        rewards_for_tasks(path, ["task_002", "task_015"])
    path.write_text(json.dumps({
        "simulations": [{"task_id": "task_002", "reward_info": {}}]
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no numeric reward"):
        rewards_for_tasks(path, ["task_002"])


def test_retired_bm25_script_points_at_the_tau_runner():
    with pytest.raises(RuntimeError, match="run_x1_full.py"):
        retired_x1_main()


def test_x4_without_a_mode_raises_and_character_means_use_actual_hits():
    old = sys.argv
    sys.argv = ["eval/run_x4_x7_context_ablation.py"]
    try:
        with pytest.raises(RuntimeError, match="does not invent"):
            x4_main()
    finally:
        sys.argv = old
    summary = summarize_retrieval_chars([
        {"full_chars": [100, 40], "section_chars": [10], "section_faithful": [True]},
        {"full_chars": [20], "section_chars": [5, 15], "section_faithful": [False, True]},
    ])
    assert summary["n_queries"] == 2
    assert summary["mean_full_doc_chars"] == (100 + 40 + 20) / 3
    assert summary["mean_section_body_chars"] == (10 + 5 + 15) / 3
    assert summary["clause_fidelity"] == 2 / 3
    assert summary["queries_short_of_10_docs"] == 2


def test_cache_dirname_includes_the_full_signature_and_not_the_experiment_name():
    shared = dict(
        agent="knowdesk_graph",
        retrieval="hybrid",
        prompt_version="v2",
        max_steps=70,
        guard_mode="1",
        task_ids=["task_002", "task_015"],
        num_trials=1,
        domain="banking_knowledge",
        language=None,
    )
    off = cache_dirname(compaction_mode="0", **shared)
    on = cache_dirname(compaction_mode="1", **shared)
    assert off == cache_dirname(compaction_mode="0", **shared)
    assert off != on
    assert "x2" not in off and "x5" not in off
    assert "_p" in off and "_ms70_" in off and "_g1_" in off and "_c0_" in off


def test_readme_renderer_embeds_sources_without_inflated_claims():
    from eval.render_readme import BANNED_PHRASES, build_readme, parse_collect_summary, verify_wiring

    root = Path(".")
    verify_wiring(root)
    summary = "tests/test_render.py::test_one\n\n1 test collected in 0.01s\n"
    assert parse_collect_summary(summary)["collected"] == 1
    manifest = {
        "exp_id": "demo",
        "git_sha": "abc",
        "timestamp": "2026-11-19T00:00:00Z",
        "model": "none",
        "llm_args": {},
        "seed": 42,
        "split_name": "demo",
        "task_ids": ["task_002"],
        "max_steps": None,
        "command": "uv run --no-sync python eval/render_readme.py",
        "metrics": {"status": "not_run"},
        "git_dirty": True,
    }
    body = "# Demo\n\n尚未运行，不填写通过率。\n"
    text = build_readme(
        root,
        pytest_summary=summary,
        manifests=[("demo_manifest.json", manifest)],
        results=[("RESULTS_DEMO.md", body)],
    )
    assert "评测边界与诚实局限（Honest Limitations）" in text
    assert body.strip() in text
    assert '"status": "not_run"' in text
    assert "1 test collected in 0.01s" in text
    assert "collected `1`" in text
    assert "不构造 `PlainAgentLoop`" in text
    assert "search_with_subagent_isolation" in text
    for phrase in BANNED_PHRASES:
        assert phrase not in text
