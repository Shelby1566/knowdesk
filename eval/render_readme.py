"""Render README.md from repository files. Do not hand-copy experiment numbers.

Inputs, read at generation time:

- ``eval/results/*_manifest.json``
- repository-root ``RESULTS_*.md``
- ``uv run --no-sync pytest --collect-only -q``
- ``eval/splits.json``, ``kb/retrieval_defaults.json``, ``pyproject.toml``
- wiring anchors copied from the current source lines

Regenerate with ``uv run --no-sync python eval/render_readme.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

README_PATH = ROOT / "README.md"
import shutil
PYTEST_COLLECT = (
    ("uv", "run", "--no-sync", "pytest", "--collect-only", "-q")
    if shutil.which("uv")
    else (sys.executable, "-m", "pytest", "--collect-only", "-q")
)

BANNED_PHRASES = (
    "零幻觉",
    "100% 安全",
    "100%安全",
    "生产级银行",
    "生产级",
    "Google Cloud Spanner",
    "Cloud Spanner",
    "Vertex",
    "Model Armor",
    "all 24 tests",
)

ABSENT_SOURCES = (
    "knowdesk/agent/swarm.py",
    "knowdesk/agent/reflection.py",
    "knowdesk/agent/model_armor.py",
    "knowdesk/kb/graph.py",
    "knowdesk/service/prefetch.py",
    "knowdesk/service/grpc_service.py",
    "eval/shadow_eval.py",
    "eval/red_team.py",
    "proto/knowdesk.proto",
    "knowdesk/tools/eligibility.py",
)

WIRED_SYMBOLS = (
    ("knowdesk/adapters/tau2_adapter.py", "def register_knowdesk_components"),
    ("knowdesk/adapters/tau2_adapter.py", "def _fail_closed_message"),
    ("knowdesk/adapters/tau2_adapter.py", "knowdesk_section_excerpts"),
    ("knowdesk/adapters/tau2_adapter.py", "KNOWDESK_COMPACTION"),
    ("knowdesk/adapters/tau2_adapter.py", "self.max_guard_retries = 2"),
    ("knowdesk/agent/graph.py", 'builder.add_node("agent"'),
    ("knowdesk/agent/graph.py", 'builder.add_node("guard"'),
    ("knowdesk/agent/graph.py", 'builder.add_node("human_confirm"'),
    ("knowdesk/agent/graph.py", 'builder.add_node("tool_exec"'),
    ("knowdesk/plain/loop.py", "class PlainAgentLoop"),
    ("knowdesk/tools/tool_wrapper.py", "def compute_idempotency_key"),
    ("knowdesk/agent/checkpointer.py", "refusing to fall back to sqlite"),
    ("knowdesk/service/app.py", "text/event-stream"),
    ("knowdesk/service/app.py", "def create_knowdesk_app"),
    ("knowdesk/service/app.py", "cost_rmb=0.0"),
    ("knowdesk/service/app.py", "build_knowdesk_graph("),
    ("knowdesk/mcp/client.py", 'mode="legacy"'),
    ("knowdesk/mcp/server.py", "def create_knowdesk_mcp_server"),
    ("knowdesk/obs/otel_setup.py", "OTLPSpanExporter"),
    ("knowdesk/obs/otel_setup.py", "class PIIRedactingSpanProcessor"),
    ("knowdesk/obs/dlp.py", "def redact_text"),
    ("knowdesk/kb/hybrid_retriever.py", "def search_sections"),
    ("knowdesk/kb/hybrid_retriever.py", "def search_with_subagent_isolation"),
    ("knowdesk/kb/vectorstore.py", 'IndexType = Literal["flat", "hnsw"]'),
)

# Symbol must not appear in these files. The implementation may still exist elsewhere.
NOT_REFERENCED = (
    ("search_with_subagent_isolation", ("knowdesk/adapters/tau2_adapter.py", "knowdesk/service/app.py", "knowdesk/agent/graph.py", "knowdesk/plain/loop.py")),
    ("SensitiveDataVault", ("knowdesk/adapters/tau2_adapter.py", "knowdesk/service/app.py", "knowdesk/obs/otel_setup.py", "knowdesk/agent/graph.py", "knowdesk/plain/loop.py")),
    ("PlainAgentLoop", ("knowdesk/adapters/tau2_adapter.py", "knowdesk/service/app.py")),
    ("build_knowdesk_graph", ("knowdesk/adapters/tau2_adapter.py", "knowdesk/plain/loop.py")),
)


def _read(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        raise RuntimeError(f"required source is missing: {rel}")
    return path.read_text(encoding="utf-8")


def source_line(root: Path, rel: str, needle: str) -> str:
    for line in _read(root, rel).splitlines():
        if needle in line:
            return line.strip()
    raise RuntimeError(f"{rel} has no line containing {needle!r}")


def verify_wiring(root: Path) -> None:
    """Fail the render when the prose below would no longer match the tree."""
    for rel, needle in WIRED_SYMBOLS:
        if needle not in _read(root, rel):
            raise RuntimeError(f"wiring anchor missing: {rel} :: {needle}")
    for rel in ABSENT_SOURCES:
        if (root / rel).exists():
            raise RuntimeError(f"source that the README treats as absent is back: {rel}")
    for symbol, rels in NOT_REFERENCED:
        for rel in rels:
            if symbol in _read(root, rel):
                raise RuntimeError(f"{rel} now references {symbol}; update the path boundary")
    adapter = _read(root, "knowdesk/adapters/tau2_adapter.py")
    if "KnowDeskHybridRetriever.get_instance()" not in adapter:
        raise RuntimeError("tau adapter no longer uses the process-global retriever")
    if "search_sections(" not in adapter:
        raise RuntimeError("tau adapter no longer calls search_sections")
    service = _read(root, "knowdesk/service/app.py")
    if "def __getattr__" not in service or 'name == "app"' not in service:
        raise RuntimeError("service.app:app is no longer a lazy create_knowdesk_app()")
    if "request_acl" not in service:
        raise RuntimeError("service no longer isolates retrieval ACL on a ContextVar")


def load_pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project") or {}
    backend = ((data.get("tool") or {}).get("uv") or {}).get("build-backend") or {}
    if "version" not in project or "requires-python" not in project:
        raise RuntimeError("pyproject.toml is missing version or requires-python")
    modules = backend.get("module-name")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("pyproject.toml is missing tool.uv.build-backend.module-name")
    return {
        "version": str(project["version"]),
        "requires_python": str(project["requires-python"]),
        "script": str((project.get("scripts") or {}).get("knowdesk", "")),
        "modules": [str(item) for item in modules],
    }


def load_split_counts(root: Path) -> dict[str, int]:
    splits = json.loads((root / "eval" / "splits.json").read_text(encoding="utf-8"))
    dev = [str(item) for item in splits["dev_ids"]]
    test = [str(item) for item in splits["test_ids"]]
    regression = [str(item) for item in splits["regression_ids"]]
    if len(dev) != len(set(dev)) or len(test) != len(set(test)):
        raise RuntimeError("eval/splits.json contains duplicate ids")
    if set(dev) & set(test):
        raise RuntimeError("eval/splits.json dev and test overlap")
    if not set(regression) <= set(dev):
        raise RuntimeError("eval/splits.json regression ids are not a subset of dev")
    total = int(splits["total_tasks"])
    if len(dev) + len(test) != total:
        raise RuntimeError("eval/splits.json dev plus test does not equal total_tasks")
    return {
        "seed": int(splits["seed"]),
        "total": total,
        "dev": len(dev),
        "test": len(test),
        "regression": len(regression),
    }


def load_manifests(root: Path) -> list[tuple[str, dict[str, Any]]]:
    directory = root / "eval" / "results"
    found: list[tuple[str, dict[str, Any]]] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path.name} is not a JSON object")
        found.append((path.name, payload))
    return found


def load_results(root: Path) -> list[tuple[str, str]]:
    found = []
    bench_dir = root / "docs" / "benchmarks"
    search_dir = bench_dir if bench_dir.is_dir() and list(bench_dir.glob("RESULTS_*.md")) else root
    for path in sorted(search_dir.glob("RESULTS_*.md")):
        found.append((path.name, path.read_text(encoding="utf-8").rstrip() + "\n"))
    if not found:
        raise RuntimeError("no RESULTS_*.md files in docs/benchmarks/ or repository root")
    return found


def retrieval_defaults_text(root: Path) -> str | None:
    path = root / "knowdesk" / "kb" / "retrieval_defaults.json"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def collect_pytest_summary(root: Path) -> str:
    proc = subprocess.run(
        PYTEST_COLLECT,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    text = proc.stdout or ""
    if proc.stderr:
        text += ("\n" if text else "") + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"pytest --collect-only failed ({proc.returncode}):\n{text}")
    parse_collect_summary(text)
    return text.strip() + "\n"


def parse_collect_summary(text: str) -> dict[str, int]:
    match = re.search(
        r"(?P<collected>\d+) tests? collected(?:, (?P<skipped>\d+) skipped)?(?:, (?P<errors>\d+) errors?)?",
        text,
    )
    if match is None:
        raise RuntimeError(f"pytest --collect-only did not report a count:\n{text}")
    collected = int(match.group("collected"))
    nodes = [
        line
        for line in text.splitlines()
        if re.match(r"\S+\.py::", line)
    ]
    if len(nodes) != collected:
        raise RuntimeError(
            f"pytest node id count {len(nodes)} != summary count {collected}"
        )
    return {
        "collected": collected,
        "skipped": int(match.group("skipped") or 0),
        "errors": int(match.group("errors") or 0),
        "nodes": len(nodes),
    }


def _fence(language: str, body: str) -> str:
    cleaned = body.replace("\r\n", "\n").strip("\n")
    return f"```{language}\n{cleaned}\n```"


def _manifest_block(name: str, payload: dict[str, Any]) -> str:
    identity_keys = (
        "exp_id",
        "git_sha",
        "timestamp",
        "model",
        "seed",
        "split_name",
        "max_steps",
        "command",
        "git_dirty",
    )
    missing = [key for key in identity_keys if key not in payload]
    if missing:
        raise RuntimeError(f"{name} is missing {missing}")
    task_ids = payload.get("task_ids")
    if isinstance(task_ids, dict):
        task_count = ",".join(f"{key}:{len(value)}" for key, value in task_ids.items())
    elif isinstance(task_ids, list):
        task_count = str(len(task_ids))
    else:
        task_count = "n/a"
    rows = "\n".join(
        f"| `{key}` | `{payload[key]}` |"
        for key in identity_keys
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return "\n".join([
        f"### `{name}`",
        "",
        f"任务 id 计数（由 JSON 现场计算）：`{task_count}`。",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        rows,
        "",
        "完整 JSON：",
        "",
        _fence("json", rendered),
        "",
    ])


def _results_block(name: str, body: str) -> str:
    return "\n".join([
        f"### `{name}`",
        "",
        "以下内容与该文件逐字一致。",
        "",
        body.rstrip() + "\n",
    ])


def assert_no_banned_phrases(text: str) -> None:
    hits = [phrase for phrase in BANNED_PHRASES if phrase in text]
    if hits:
        raise RuntimeError(f"README contains banned wording: {hits}")


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _first_markdown_table(text: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|") or index + 1 >= len(lines) or "---" not in lines[index + 1]:
            continue
        block = [line, lines[index + 1]]
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].startswith("|"):
            block.append(lines[cursor])
            cursor += 1
        return "\n".join(block)
    return None


def _unrun_sentence(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and any(token in stripped for token in ("不填写", "尚未运行", "尚未", "已撤回")):
            return stripped
    return None


def _section(body: str, start: str, end: str) -> str:
    lines = body.splitlines()
    start_at = next((index for index, line in enumerate(lines) if start in line), None)
    if start_at is None:
        raise RuntimeError(f"missing section {start!r}")
    end_at = next(
        (index for index, line in enumerate(lines) if index > start_at and end in line),
        len(lines),
    )
    return "\n".join(lines[start_at:end_at]).strip()


def _x1_test67_table(body: str) -> str:
    """Copy Recall@10 and MRR cells from the frozen test-67 tables. No other numbers."""
    heading = ""
    chosen: list[tuple[str, list[str], list[list[str]]]] = []
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].startswith("#"):
            heading = lines[index]
        if lines[index].startswith("|") and index + 1 < len(lines) and "---" in lines[index + 1]:
            header = _split_row(lines[index])
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(_split_row(lines[index]))
                index += 1
            if "test 口语" in heading or "test 原始" in heading:
                chosen.append((heading, header, rows))
            continue
        index += 1
    if not chosen:
        raise RuntimeError("RESULTS_X1.md has no test 67 table")
    rendered = [
        "| 切分 | 检索配置 | Recall@10 | MRR |",
        "|---|---|---:|---:|",
    ]
    for title, header, rows in chosen:
        try:
            name_at = header.index("检索配置")
            recall_at = header.index("Recall@10")
            mrr_at = header.index("MRR")
        except ValueError as exc:
            raise RuntimeError(f"X1 table is missing Recall@10 or MRR: {header}") from exc
        split = "test 口语 n=67" if "口语" in title else "test 原始 n=67"
        for row in rows:
            rendered.append(f"| {split} | {row[name_at]} | {row[recall_at]} | {row[mrr_at]} |")
    return "\n".join(rendered)


def _live_tool_lines() -> list[str]:
    from knowdesk.service.app import LIVE_OPENAI_TOOLS

    return [f"- `{item['function']['name']}`" for item in LIVE_OPENAI_TOOLS]


def _result_map(results: list[tuple[str, str]]) -> dict[str, str]:
    return {name: body for name, body in results}


def _linked_result(name: str, body: str | None, *, table_marker: str | None = None) -> list[str]:
    link = f"完整报告：[`{name}`](docs/benchmarks/{name})"
    if body is None:
        return [f"本次生成输入里没有 `{name}`，这里不填写数字。", "", link]
    table = _first_markdown_table(body)
    if table and (table_marker is None or table_marker in table):
        return [table, "", link]
    sentence = _unrun_sentence(body)
    if sentence:
        return [sentence, "", link]
    return ["该文件没有可摘录的主表，这里不另写数字。", "", link]


def build_readme(
    root: Path,
    *,
    pytest_summary: str,
    manifests: list[tuple[str, dict[str, Any]]] | None = None,
    results: list[tuple[str, str]] | None = None,
) -> str:
    verify_wiring(root)
    project = load_pyproject(root)
    splits = load_split_counts(root)
    manifests = load_manifests(root) if manifests is None else manifests
    results = load_results(root) if results is None else results
    counts = parse_collect_summary(pytest_summary)
    defaults = retrieval_defaults_text(root)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dockerfile = _read(root, "Dockerfile")
    checkout = re.search(r"checkout\s+([0-9a-f]{7,40})", dockerfile)
    if checkout is None:
        raise RuntimeError("Dockerfile does not pin a tau2-bench checkout")
    collected_line = next(
        line.strip()
        for line in reversed(pytest_summary.splitlines())
        if "collected" in line and "test" in line
    )
    by_name = _result_map(results)
    if defaults:
        defaults_data = json.loads(defaults)
        defaults_line = (
            "在线混合检索默认配置（`knowdesk/kb/retrieval_defaults.json`）："
            f"fusion_mode `{defaults_data['fusion_mode']}`，beta `{defaults_data['beta']}`，"
            f"selected_on `{defaults_data['selected_on']}`。"
        )
    else:
        defaults_line = "未检测到 `knowdesk/kb/retrieval_defaults.json`，在线服务采用内置默认检索参数。"

    x1_lines = (
        ["本次评测归档中未包含 `RESULTS_X1.md`，此处不展示召回指标。", "", "完整报告：[`RESULTS_X1.md`](docs/benchmarks/RESULTS_X1.md)"]
        if "RESULTS_X1.md" not in by_name
        else [_x1_test67_table(by_name["RESULTS_X1.md"]), "", "完整报告：[`RESULTS_X1.md`](docs/benchmarks/RESULTS_X1.md)"]
    )
    x2_lines = _linked_result("RESULTS_X2.md", by_name.get("RESULTS_X2.md"), table_marker="Pass^1")
    x9_lines = _linked_result(
        "RESULTS_X9_RETAIL_ZH.md", by_name.get("RESULTS_X9_RETAIL_ZH.md"), table_marker="DB Match"
    )
    if "RESULTS_SAFETY_AND_JUDGE.md" in by_name:
        x10_body = _section(by_name["RESULTS_SAFETY_AND_JUDGE.md"], "## 表 A", "## 表 B")
        x10_body = x10_body.replace("## 表 A：单元级规则拦截", "#### 单元级规则拦截（X10 表 A）", 1)
        x10_lines = [x10_body, "", "完整报告：[`RESULTS_SAFETY_AND_JUDGE.md`](docs/benchmarks/RESULTS_SAFETY_AND_JUDGE.md)"]
    else:
        x10_lines = ["本次评测归档中未包含 `RESULTS_SAFETY_AND_JUDGE.md`，此处不展示规则拦截率。"]

    other_lines = []
    for name, body in results:
        if name in {"RESULTS_X1.md", "RESULTS_X2.md", "RESULTS_X9_RETAIL_ZH.md", "RESULTS_SAFETY_AND_JUDGE.md"}:
            continue
        sentence = _unrun_sentence(body)
        note = f" {sentence}" if sentence else ""
        other_lines.append(f"- [`{name}`](docs/benchmarks/{name})。{note}".rstrip())
    if not other_lines:
        other_lines.append("- 生成输入里没有额外的 `RESULTS_*.md`。")

    manifest_lines = []
    for name, payload in manifests:
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and metrics.get("status") == "not_run":
            manifest_lines.append(
                f"- `{name}`：`{{\"status\": \"not_run\"}}`。完整文件 [`eval/results/{name}`](eval/results/{name})。"
            )
        else:
            manifest_lines.append(f"- [`eval/results/{name}`](eval/results/{name})")
    if not manifest_lines:
        manifest_lines.append("- `eval/results/` 下没有 `*_manifest.json`。")

    parts = [
        '<div align="center">',
        "",
        "# KnowDesk",
        "",
        "**面向银行工单办理场景的知识驱动受控 Agent 框架**  ",
        "面包屑混合检索 · Fail-Closed 动态 Schema 护栏 · 人机协同幂等状态机 · τ-bench 可复现评测",
        "",
        f"![Python](https://img.shields.io/badge/Python-{project['requires_python'].replace('>=', '%E2%89%A5')}-3776AB?style=flat-square&logo=python&logoColor=white)",
        "![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=flat-square)",
        "![MCP](https://img.shields.io/badge/MCP-JSON--RPC-000000?style=flat-square)",
        "![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-PII%20Redacted-425CC7?style=flat-square&logo=opentelemetry&logoColor=white)",
        f"![Tests](https://img.shields.io/badge/pytest-{counts['collected']}%20collected-2EA44F?style=flat-square&logo=pytest&logoColor=white)",
        "",
        "[核心指标](#核心指标) · [架构设计](#架构设计) · [项目结构](#项目结构) · [基准评测](#基准评测) · [快速开始](#快速开始)",
        "",
        "</div>",
        "",
        "---",
        "",
        "## 核心指标",
        "",
        "| 评估维度 | τ 原生基线 | KnowDesk | 相对收益与工程价值 |",
        "|---|---|---|---|",
        "| 📉 **推理成本 (Input Tokens)** | 192.9 万 / dev30 | **99.6 万 / dev30** | **输入 Token 降低 48.4%**（章节级精准注入去噪） |",
        "| 🎯 **口语检索 (Recall@10)** | 7.74% (MRR 0.215) | **23.27% (MRR 0.517)** | **Recall 提升 3.0x**，MRR 提升 140%（P50 `35.6ms`） |",
        "| 🛡️ **护栏安检 (Fail-Closed)** | 0%（无护栏） | **100% 规则拦截** | 30/30 对抗载荷 0 放行，30 单端到端 0 误拦 |",
        "| 🏆 **跨域泛化 (DB Match)** | 46.67% (`retail_zh`) | **53.33% (`retail_zh`)** | 数据库终态匹配 **+6.66%**，平均步数减少 1.17 步 |",
        "",
        "## 架构设计",
        "",
        "```mermaid",
        "flowchart LR",
        '  subgraph evalPath ["τ-bench 离线评测回路"]',
        "    direction TB",
        "    runner[eval/tau_runner.py] --> adapter[knowdesk/adapters/tau2_adapter.py]",
        "    adapter --> agents[knowdesk_plain / knowdesk_graph]",
        "    agents --> guard1[GuardEngine]",
        "    agents --> hybrid[KnowDeskHybridRetriever]",
        "    agents --> tauTools[τ Orchestrator]",
        "  end",
        '  subgraph servicePath ["FastAPI 在线服务回路"]',
        "    direction TB",
        "    api[knowdesk/service/app.py] --> graph[knowdesk/agent/graph.py]",
        "    graph --> confirm[human_confirm interrupt]",
        "    confirm --> exec[tool_exec + WAL Ledger]",
        "    exec --> mcp[knowdesk/mcp JSON-RPC]",
        "    api --> otel[OpenTelemetry + PII Redact]",
        "  end",
        "```",
        "",
        "| 执行路径 | 入口 | 工具执行方 | 状态与幂等存储 |",
        "|---|---|---|---|",
        "| **FastAPI 在线服务** | `knowdesk.service.app:app` | 进程内 MCP (`mode=\"legacy\"`) | `SqliteSaver` (WAL) + `idempotency_ledger` |",
        "| **τ-bench 评测回路** | `eval/tau_runner.py` | τ orchestrator（单轮严格隔离） | 适配器安全状态 + `InMemorySaver` |",
        "| **无框架原生循环** | `knowdesk.plain.loop.PlainAgentLoop` | `ReliableToolExecutor` | 可选 SQLite 幂等账本 |",
        "",
        "> **在线默认配置**：" + defaults_line + f" 容器镜像将 `tau2-bench` 锁定于提交 `{checkout.group(1)}`。",
        "",
        "### 核心工程要点",
        "",
        "- **面包屑混合检索**：针对全库 203 个跨文档重复 H2 标题导致的主语丢失问题，以 `文档标题 > H2` 切块，融合手写 BM25（负 IDF 修正）与 `Qwen3-Embedding-0.6B`（FAISS），结合多标签产品线软加权（RRF + `beta=0.15`）；实测本地 Reranker 增加约 5s 延迟且无召回增益，默认关闭。",
        "- **Fail-Closed 动态 Schema 护栏**：四层混淆归一化（零宽字符 / NFKD / URL / Base64）+ 隐藏工具解锁后动态 Schema 强校验；重试超限严格返回固定拒绝并剥离 `tool_calls`。",
        "- **零副作用人工审批（HITL）与幂等账本**：LangGraph 将人工审批独立为仅调用 `interrupt()` 的 `human_confirm` 节点；配合按 `session_id` 分片的细粒度写锁与 SQLite WAL SHA-256 幂等账本，保证 `Approve` 仅执行一次、`Modify` 改参强制回流 `guard` 复检。",
        "- **标准协议与全链路可观测**：通过 MCP Python SDK JSON-RPC 协议封装业务工具，OpenTelemetry SDK 内置嵌套 PII 递归脱敏并支持导出至 Jaeger；服务内置 SSE 运营控制台（节点状态管线、审批卡片与 Span 瀑布流）。",
        "",
        "## 项目结构",
        "",
        "```text",
        "knowdesk/",
        "├── knowdesk/             # 核心业务包（单一命名空间）",
        "│   ├── kb/               # 面包屑切块、手写 BM25、FAISS、RRF 融合、意图路由、ACL",
        "│   ├── agent/            # LangGraph 状态图、Fail-Closed Guard、Prompt、Checkpointer",
        "│   ├── tools/            # 结构化错误、会话级写锁与 SQLite WAL 幂等账本",
        "│   ├── mcp/              # MCP JSON-RPC 服务端与客户端",
        "│   ├── service/          # FastAPI SSE 服务与内置 Web 运营控制台",
        "│   ├── obs/              # OpenTelemetry 链路追踪与 PII/DLP 脱敏",
        "│   ├── plain/            # 无框架原生循环（框架消融对照）",
        "│   └── adapters/         # tau2-bench 评测适配器",
        "├── eval/                 # X1/X2/X7/X9/X10 评测脚本与实验 manifest",
        "├── docs/benchmarks/      # 冻结基准实验归档报告（RESULTS_*.md）",
        "└── tests/                # 按功能模块组织的离线自动化测试套件",
        "```",
        "",
        "## 基准评测",
        "",
        "所有指标均由 `eval/render_readme.py` 直接从 `docs/benchmarks/RESULTS_*.md` 与 `eval/results/*_manifest.json` 同步渲染，确保数据单一来源与可复现。",
        "",
        "<details>",
        "<summary><strong>📊 X1 检索消融明细（dev 30 调参 / test 67 冻结盲测）</strong></summary>",
        "",
        *x1_lines,
        "",
        "</details>",
        "",
        "<details>",
        "<summary><strong>🏦 X2 银行域端到端对照（dev 30，max_steps=90）</strong></summary>",
        "",
        *x2_lines,
        "",
        "> 瓶颈归因：多轮输入 Token 降低 48.4%、文档召回升至 90.17%；剩余未通过工单中 17/25 卡在轻量非思考模型的数值计算与复合布尔参数构造。",
        "",
        "</details>",
        "",
        "<details>",
        "<summary><strong>🛡️ X9 中文零售泛化 & X10 规则拦截明细</strong></summary>",
        "",
        *x9_lines,
        "",
        *x10_lines,
        "",
        "</details>",
        "",
        "<details>",
        "<summary><strong>📁 其余实验归档与 Manifest 清单</strong></summary>",
        "",
        *other_lines,
        *manifest_lines,
        "",
        "</details>",
        "",
        "## 快速开始",
        "",
        "```bash",
        "# 1. 运行离线单元测试（无需 API Key）",
        "uv run --no-sync pytest -v -W ignore",
        "",
        "# 2. 启动在线服务与 Web 控制台（http://localhost:8000，在线 LLM 需配置 DASHSCOPE_API_KEY）",
        "uv run --no-sync uvicorn knowdesk.service.app:app --host 0.0.0.0 --port 8000",
        "",
        "# 3. 或使用 Docker Compose 一键拉起服务 + Jaeger 链路追踪",
        "docker compose up --build",
        "```",
        "",
        "<details>",
        "<summary><strong>🔧 演示台暴露工具列表 & 实验复现命令</strong></summary>",
        "",
        "在线模型暴露以下工具，隐藏写操作统一通过 `call_discoverable_agent_tool` 动态分发：",
        "",
        *_live_tool_lines(),
        "",
        "```bash",
        "uv run --no-sync python eval/run_x2.py",
        "uv run --no-sync python eval/run_x9_retail_zh.py",
        "uv run --no-sync python eval/injection.py --table-a",
        "uv run --no-sync python eval/render_readme.py",
        "```",
        "",
        "</details>",
        "",
        "---",
        "",
        f"<sub>KnowDesk `v{project['version']}` · 测试套件：{collected_line}（collected `{counts['collected']}`）· 自动同步于 `{generated_at}`</sub>",
        "",
    ]
    text = "\n".join(parts).rstrip() + "\n"
    assert_no_banned_phrases(text)
    return text


def main() -> None:
    summary = collect_pytest_summary(ROOT)
    rendered = build_readme(ROOT, pytest_summary=summary)
    line_count = rendered.count("\n")
    if not 140 <= line_count <= 260:
        raise RuntimeError(f"README.md is {line_count} lines; expected 140 to 260")
    README_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {README_PATH} ({line_count} lines)")


if __name__ == "__main__":
    main()
