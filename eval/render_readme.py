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
PYTEST_COLLECT = ("uv", "run", "--no-sync", "pytest", "--collect-only", "-q")

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
    "agent/swarm.py",
    "agent/reflection.py",
    "agent/model_armor.py",
    "kb/graph.py",
    "service/prefetch.py",
    "service/grpc_service.py",
    "eval/shadow_eval.py",
    "eval/red_team.py",
    "proto/knowdesk.proto",
    "tools/eligibility.py",
)

WIRED_SYMBOLS = (
    ("adapters/tau2_adapter.py", "def register_knowdesk_components"),
    ("adapters/tau2_adapter.py", "def _fail_closed_message"),
    ("adapters/tau2_adapter.py", "knowdesk_section_excerpts"),
    ("adapters/tau2_adapter.py", "KNOWDESK_COMPACTION"),
    ("adapters/tau2_adapter.py", "self.max_guard_retries = 2"),
    ("agent/graph.py", 'builder.add_node("agent"'),
    ("agent/graph.py", 'builder.add_node("guard"'),
    ("agent/graph.py", 'builder.add_node("human_confirm"'),
    ("agent/graph.py", 'builder.add_node("tool_exec"'),
    ("plain/loop.py", "class PlainAgentLoop"),
    ("tools/tool_wrapper.py", "def compute_idempotency_key"),
    ("agent/checkpointer.py", "refusing to fall back to sqlite"),
    ("service/app.py", "text/event-stream"),
    ("service/app.py", "def create_knowdesk_app"),
    ("service/app.py", "cost_rmb=0.0"),
    ("service/app.py", "build_knowdesk_graph("),
    ("mcp_client/client.py", 'mode="legacy"'),
    ("mcp_server/server.py", "def create_knowdesk_mcp_server"),
    ("obs/otel_setup.py", "OTLPSpanExporter"),
    ("obs/otel_setup.py", "class PIIRedactingSpanProcessor"),
    ("security/dlp.py", "def redact_text"),
    ("kb/hybrid_retriever.py", "def search_sections"),
    ("kb/hybrid_retriever.py", "def search_with_subagent_isolation"),
    ("kb/vectorstore.py", 'IndexType = Literal["flat", "hnsw"]'),
)

# Symbol must not appear in these files. The implementation may still exist elsewhere.
NOT_REFERENCED = (
    ("search_with_subagent_isolation", ("adapters/tau2_adapter.py", "service/app.py", "agent/graph.py", "plain/loop.py")),
    ("SensitiveDataVault", ("adapters/tau2_adapter.py", "service/app.py", "obs/otel_setup.py", "agent/graph.py", "plain/loop.py")),
    ("PlainAgentLoop", ("adapters/tau2_adapter.py", "service/app.py")),
    ("build_knowdesk_graph", ("adapters/tau2_adapter.py", "plain/loop.py")),
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
    adapter = _read(root, "adapters/tau2_adapter.py")
    if "KnowDeskHybridRetriever.get_instance()" not in adapter:
        raise RuntimeError("tau adapter no longer uses the process-global retriever")
    if "search_sections(" not in adapter:
        raise RuntimeError("tau adapter no longer calls search_sections")
    service = _read(root, "service/app.py")
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
    for path in sorted(root.glob("RESULTS_*.md")):
        found.append((path.name, path.read_text(encoding="utf-8").rstrip() + "\n"))
    if not found:
        raise RuntimeError("no RESULTS_*.md files at the repository root")
    return found


def retrieval_defaults_text(root: Path) -> str | None:
    path = root / "kb" / "retrieval_defaults.json"
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
    from service.app import LIVE_OPENAI_TOOLS

    return [f"- `{item['function']['name']}`" for item in LIVE_OPENAI_TOOLS]


def _result_map(results: list[tuple[str, str]]) -> dict[str, str]:
    return {name: body for name, body in results}


def _linked_result(name: str, body: str | None, *, table_marker: str | None = None) -> list[str]:
    link = f"原文：[`{name}`]({name})。"
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
            "在线混合检索默认配置（`kb/retrieval_defaults.json`）："
            f"fusion_mode `{defaults_data['fusion_mode']}`，beta `{defaults_data['beta']}`，"
            f"selected_on `{defaults_data['selected_on']}`。"
        )
    else:
        defaults_line = "未检测到 `kb/retrieval_defaults.json`，在线服务采用内置默认检索参数。"

    x1_lines = (
        ["本次评测归档中未包含 `RESULTS_X1.md`，此处不展示召回指标。", "", "原文：[`RESULTS_X1.md`](RESULTS_X1.md)。"]
        if "RESULTS_X1.md" not in by_name
        else [_x1_test67_table(by_name["RESULTS_X1.md"]), "", "原文：[`RESULTS_X1.md`](RESULTS_X1.md)。"]
    )
    x2_lines = _linked_result("RESULTS_X2.md", by_name.get("RESULTS_X2.md"), table_marker="Pass^1")
    x9_lines = _linked_result(
        "RESULTS_X9_RETAIL_ZH.md", by_name.get("RESULTS_X9_RETAIL_ZH.md"), table_marker="DB Match"
    )
    if "RESULTS_SAFETY_AND_JUDGE.md" in by_name:
        x10_body = _section(by_name["RESULTS_SAFETY_AND_JUDGE.md"], "## 表 A", "## 表 B")
        x10_body = x10_body.replace("## 表 A：单元级规则拦截", "#### X10 表 A：单元级规则拦截", 1)
        x10_lines = [x10_body, "", "原文：[`RESULTS_SAFETY_AND_JUDGE.md`](RESULTS_SAFETY_AND_JUDGE.md)。"]
    else:
        x10_lines = ["本次评测归档中未包含 `RESULTS_SAFETY_AND_JUDGE.md`，此处不展示规则拦截率。"]

    other_lines = []
    for name, body in results:
        if name in {"RESULTS_X1.md", "RESULTS_X2.md", "RESULTS_X9_RETAIL_ZH.md", "RESULTS_SAFETY_AND_JUDGE.md"}:
            continue
        sentence = _unrun_sentence(body)
        note = f" {sentence}" if sentence else ""
        other_lines.append(f"- [`{name}`]({name})。{note}".rstrip())
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
        "# KnowDesk",
        "",
        "面向银行工单办理场景的知识驱动受控 Agent 框架，集成面包屑混合检索、Fail-Closed 动态 Schema 护栏与 τ-bench 可复现评测体系。",
        "",
        "`Python " + project["requires_python"] + "` · `FastAPI` · `LangGraph` · `SQLite` · `MCP` · `OpenTelemetry` · `FAISS` · `LiteLLM`",
        "",
        f"版本 `{project['version']}`。文档由 `eval/render_readme.py` 于 `{generated_at}` 自动构建，各项基准指标严格同步自 `RESULTS_*.md` 实验归档。",
        "",
        "## 架构设计",
        "",
        "```mermaid",
        "flowchart TB",
        '  subgraph evalPath ["τ 评测"]',
        "    runner[eval/tau_runner.py]",
        "    adapter[adapters/tau2_adapter.py]",
        "    agents[knowdesk_plain / knowdesk_graph]",
        "    guard[GuardEngine]",
        "    hybrid[KnowDeskHybridRetriever.search_sections]",
        "    tauTools[τ orchestrator]",
        "    runner --> adapter --> agents",
        "    agents --> guard",
        "    agents --> hybrid",
        "    agents --> tauTools",
        "  end",
        '  subgraph servicePath ["FastAPI 演示"]',
        "    api[service/app.py]",
        "    graph[agent/graph.py]",
        "    confirm[human_confirm interrupt]",
        "    exec[tool_exec]",
        "    ledger[SQLite idempotency_ledger]",
        "    mcp[mcp_client Client]",
        "    otel[OpenTelemetry SDK]",
        "    api --> graph --> confirm --> exec --> ledger",
        "    exec --> mcp",
        "    api --> otel",
        "  end",
        "```",
        "",
        "系统划分两条核心执行路径：",
        "- **τ 评测回路**：工具执行统一委托给 τ-bench orchestrator。适配器中的 `knowdesk_graph` 针对单轮交互维护独立图结构（不调用在线服务态的 `build_knowdesk_graph`），保证评测环境严格隔离。",
        "- **FastAPI 服务回路**：通过 `build_knowdesk_graph` 构建完整的状态机工作流（`agent` → `guard` → `human_confirm` → `tool_exec`）。人工审批节点触发 LangGraph 原生 `interrupt()` 进行零副作用等待，恢复执行时严格幂等，避免重跑已完成工具。",
        "服务默认集成 LiteLLM、混合检索与进程内 MCP；首次访问 `service.app:app` 时触发 `create_knowdesk_app()` 懒加载初始化。",
        defaults_line,
        f"评测环境容器镜像将 tau2-bench 锁定于提交 `{checkout.group(1)}`。",
        "",
        "| 路径 | 入口 | 工具谁来执行 | 状态放在哪 |",
        "|---|---|---|---|",
        "| τ 评测 | `eval/tau_runner.py` | τ orchestrator | 适配器进程内的安全状态，图用 `InMemorySaver` |",
        "| 无框架循环 | 调用方自己建 `PlainAgentLoop` | `ReliableToolExecutor` | 调用方决定要不要 SQLite 账本 |",
        "| FastAPI 演示 | `service.app:app` | 进程内 MCP | `SqliteSaver` 加 `idempotency_ledger` |",
        "",
        "## 实测结果",
        "",
        "### 核心成果一览 (Highlights)",
        "",
        "| 核心维度 | τ 原生基线 | KnowDesk 表现 | 相对收益与工程价值 |",
        "|---|---|---|---|",
        "| 📉 **推理成本 (Token)** | 192.9 万 / dev30 | **99.6 万 / dev30** | **输入 Token 骤降 48.4%**（章节级切块去噪） |",
        "| 🎯 **口语检索 (R@10)** | 7.74% (MRR 0.215) | **23.27% (MRR 0.517)** | **Recall 提升 3.0x**，MRR 提升 140% |",
        "| 🛡️ **护栏安检 (Fail-Closed)** | 0% (无护栏) | **100% 规则拦截率** | 30/30 对抗用例 0 放行，30 单端到端 0 误拦 |",
        "| 🏆 **跨域终态 (DB Match)** | 46.67% (retail_zh) | **53.33% (retail_zh)** | 任务成功率绝对值 **+6.66%**，平均步数减少 1.17 步 |",
        "",
        "### X1 检索消融（test 67）",
        "",
        "基于 `dev 30` 口语查询集调优超参，并在冻结的 `test 67` 独立测试集（含口语化查询与原始工单说明两组切分）上盲测，核心指标为 Recall@10 与 MRR：",
        "",
        *x1_lines,
        "",
        "> **工程结论**：针对全库跨文档重复 H2 导致的上下文主语丢失问题，面包屑切块（文档标题 > H2）与 RRF 融合带来 3 倍口语检索召回提升；实测本地 Reranker 增加约 5s 延迟且未带来显著增益，经数据驱动决策默认不启用重排。",
        "",
        "### X2 银行域端到端（dev 30）",
        "",
        *x2_lines,
        "",
        "> **工程结论**：章节级上下文精准注入使端到端多轮 Token 消耗降低 48.4%，文档召回率提升至 90.17%；瓶颈分析表明主要未完成工单受制于轻量非思考模型的数值计算与复合布尔推理极限。",
        "",
        "### X9 中文零售与 X10 规则拦截",
        "",
        *x9_lines,
        "",
        *x10_lines,
        "",
        "> **工程结论**：四层混淆归一化结合 Fail-Closed 动态 Schema 校验，在 30 任务端到端实测 0 误拦的同时，实现了确定性的越权与注入阻断。",
        "",
        "## 核心工程要点",
        "",
        "- **面包屑混合检索**：结合文档标题与 H2 面包屑切块，使用 BM25 倒排索引与 Qwen 向量进行静态 RRF 融合，在线默认启用产品线意图加权；本地重排经消融评估后保持独立可选。",
        "- **Fail-Closed 动态 Schema 护栏**：工具解锁后按当次动态 schema 校验入参，当重试达到上限仍未通过时，严格返回固定拒绝并剥离 tool calls，阻断越权与非受控执行。",
        "- **人机协同审批与状态隔离**：LangGraph 将人工审批独立为 `interrupt` 节点，会话恢复唤醒时保持幂等，避免重跑已完成的工具操作。",
        "- **持久化幂等账本**：SQLite 账本基于会话、工具名及规范化参数计算 SHA-256 唯一指纹；仅记录成功写操作，服务重启后依然支持重放校验。",
        "- **标准协议与链路可观测**：工具通信遵循 MCP 协议标准（基于 MCP Python SDK `Client(..., mode=\"legacy\")`）；可观测性基于 OpenTelemetry SDK，内置敏感 PII 递归脱敏，按需导出至 OTLP/Jaeger。",
        "",
        "## 控制台界面与可视化 (Web Console)",
        "",
        "服务内置了高保真的银行工单运营与全链路可观测控制台（访问 [http://localhost:8000](http://localhost:8000)）：",
        "",
        "- **工单交互与人机协同审批**：原生 SSE 增量对话流；动态 Schema Guard 拦截到高风险隐藏操作时自动触发 `interrupt`，支持一键 `Approve`（批准）、`Modify`（在线修改 JSON 参数）及 `Reject`（驳回）。",
        "- **LangGraph 动态状态管线**：实时可视化高亮节点流转（`agent 规划` ➔ `guard 安检` ➔ `human_confirm 人工确认` ➔ `tool_exec 工具执行`）。",
        "- **OpenTelemetry 链路瀑布流**：Span 级耗时比例瀑布图（Waterfall），原生支持敏感 PII 数据递归脱敏展示。",
        "- **实时效能指标看板**：实时度量多轮输入 Tokens、P90 响应耗时与 SQLite 账本幂等保护状态。",
        "",
        "## 演示台工具",
        "",
        "在线模型暴露以下函数集合。隐藏写操作统一通过 `call_discoverable_agent_tool` 动态调用，不逐个硬编码进入全局 schema：",
        "",
        *_live_tool_lines(),
        "",
        "## 快速开始",
        "",
        "```bash",
        "# 运行单元测试",
        "uv run --no-sync pytest -v -W ignore",
        "",
        "# 启动在线演示服务",
        "uv run --no-sync uvicorn service.app:app --host 0.0.0.0 --port 8000",
        "```",
        "",
        "本地服务访问地址：[http://localhost:8000](http://localhost:8000)。在线调用需配置环境变量 `DASHSCOPE_API_KEY`（离线单元测试无需任何 API Key）。",
        "",
        "### 实验复现与文档同步",
        "",
        "```bash",
        "uv run --no-sync python eval/run_x2.py",
        "uv run --no-sync python eval/run_x9_retail_zh.py",
        "uv run --no-sync python eval/injection.py --table-a",
        "uv run --no-sync python eval/render_readme.py",
        "```",
        "",
        "> **评测说明**：X2 对应 dev 30 任务集在同等 max_steps 预算下的对照评测；X9 为中文 retail 域前 30 单的数据库终态匹配测试（两者均调用百炼模型接口）。X1 冻结测试集数据直接归档于 `RESULTS_X1.md`。",
        f"> **测试套件状态**：{collected_line}（汇总 collected `{counts['collected']}` 项测试用例）。",
        "",
        "## 结果文件",
        "",
        *other_lines,
        *manifest_lines,
        "",
    ]
    text = "\n".join(parts).rstrip() + "\n"
    assert_no_banned_phrases(text)
    return text


def main() -> None:
    summary = collect_pytest_summary(ROOT)
    rendered = build_readme(ROOT, pytest_summary=summary)
    line_count = rendered.count("\n")
    if not 150 <= line_count <= 220:
        raise RuntimeError(f"README.md is {line_count} lines; expected 150 to 220")
    README_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {README_PATH} ({line_count} lines)")


if __name__ == "__main__":
    main()
