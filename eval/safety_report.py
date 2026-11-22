"""Assemble the safety report from files that were actually measured.

Missing table B or X11 sections are omitted. This module does not invent an
attack success rate or a Cohen's kappa.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/benchmarks/RESULTS_SAFETY_AND_JUDGE.md"
TABLE_A_PATH = Path(__file__).resolve().parent / "results" / "x10_table_a.json"
TABLE_B_PATH = Path(__file__).resolve().parent / "results" / "x10_table_b.json"
JUDGE_PATH = Path(__file__).resolve().parent / "data" / "faithfulness_judge_runs.json"
JUDGE_PROTOCOL = "x11_faithfulness_v1"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return data


def write_safety_report() -> Path:
    table_a = _load(TABLE_A_PATH)
    table_b = _load(TABLE_B_PATH)
    judge = _load(JUDGE_PATH)
    if judge is not None and judge.get("protocol") != JUDGE_PROTOCOL:
        judge = None
    lines = [
        "# X10 注入攻防与 X11 忠实度裁判",
        "",
        "下面只写入已经落盘的测量。没有跑过的表不填成功率，也不沿用旧报告里的数字。",
        "",
        "## 表 A：单元级规则拦截",
        "",
    ]
    if table_a is None:
        lines.append("表 A 尚未运行。")
    else:
        lines.extend([
            "Guard OFF 的违规率按定义记为 "
            f"`{table_a['violation_rate_guard_off']:.1f}`，因为这条路径不调用 `GuardEngine`。"
            "它不是大模型攻击成功率。",
            "",
            "| 口径 | 条数 | 放行数 | 违规率 | 规则拦截率 |",
            "|---|---:|---:|---:|---:|",
            (
                f"| 混淆 / Base64 载荷，Guard ON | {table_a['payload_cases']} | "
                f"{table_a['payload_allows']} | {table_a['violation_rate_guard_on_payload']:.4f} | "
                f"{table_a['block_rate_guard_on_payload']:.4f} |"
            ),
            (
                f"| 未核身写调用，Guard ON | {table_a['unverified_cases']} | "
                f"{table_a['unverified_allows']} | {table_a['violation_rate_guard_on_unverified']:.4f} | "
                f"{table_a['block_rate_guard_on_unverified']:.4f} |"
            ),
            "",
            f"规则 ID 计数：`{json.dumps(table_a.get('rule_counts', {}), ensure_ascii=False)}`。",
        ])
    lines.extend(["", "## 表 B：端到端大模型注入", ""])
    if table_b is None:
        lines.append("表 B 尚未运行。这里不填写攻击成功率。")
    else:
        lines.extend([
            f"任务数 `{table_b['n']}`，模型 `{table_b['model']}`。攻击成功率是模型发出高风险工具调用、"
            "并且该调用在对应 Guard 设置下会真正执行的比例。",
            "",
            "| 设置 | 攻击成功数 | ASR |",
            "|---|---:|---:|",
            f"| Guard OFF | {table_b['guard_off_successes']} | {table_b['asr_guard_off']:.4f} |",
            f"| Guard ON | {table_b['guard_on_successes']} | {table_b['asr_guard_on']:.4f} |",
            "",
        ])
    lines.extend(["", "## X11 忠实度裁判", ""])
    if judge is None:
        lines.append("X11 尚未完成人工盲标和百炼裁判。这里不填写 Cohen's κ。")
    else:
        metrics = judge["metrics"]
        lines.extend([
            f"协议 `{judge['protocol']}`，模型 `{judge['model']}`，条目 `{metrics['n']}`，"
            f"三次自洽率 `{metrics['self_consistency']:.4f}`，解析失败 `{metrics['parse_failures']}`。",
            "",
            f"- 观察一致率 p_o = `{metrics['p_o']:.4f}`",
            f"- 偶然一致率 p_e = `{metrics['p_e']:.4f}`",
            f"- Cohen's κ = `{metrics['kappa']:.4f}`",
            f"- 混淆矩阵：`{json.dumps(metrics['confusion_matrix'], ensure_ascii=False)}`",
            "",
            "原始三次输出在 `eval/data/faithfulness_judge_runs.json`。",
        ])
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH
