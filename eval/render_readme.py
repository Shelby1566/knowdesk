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
            "在线默认读自 `kb/retrieval_defaults.json`："
            f"fusion_mode `{defaults_data['fusion_mode']}`，beta `{defaults_data['beta']}`，"
            f"selected_on `{defaults_data['selected_on']}`。"
        )
    else:
        defaults_line = "`kb/retrieval_defaults.json` 不在仓库里。在线路径会使用代码里的内置默认。"

    x1_lines = (
        ["本次生成输入里没有 `RESULTS_X1.md`，这里不填写召回。", "", "原文：[`RESULTS_X1.md`](RESULTS_X1.md)。"]
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
        x10_lines = ["本次生成输入里没有 `RESULTS_SAFETY_AND_JUDGE.md`，这里不填写规则拦截率。"]

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
        "KnowDesk 是银行工单代理的执行与评测骨架：面包屑混合检索、失败关闭的动态 Schema Guard，以及可复现的 τ-bench 对照。",
        "",
        "`Python " + project["requires_python"] + "` · `FastAPI` · `LangGraph` · `SQLite` · `MCP` · `OpenTelemetry` · `FAISS` · `LiteLLM`",
        "",
        f"包版本 `{project['version']}`。本文由 `eval/render_readme.py` 在 `{generated_at}` 生成。表里的数字只从当时读到的 `RESULTS_*.md` 单元格抄出。",
        "",
        "## 架构",
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
        "评测路径里的工具由 τ orchestrator 执行。`knowdesk_graph` 在适配器里另编一张每回合的图，不调用 `build_knowdesk_graph`。",
        "演示路径进入 `build_knowdesk_graph`：`agent`、`guard`、`human_confirm`、`tool_exec`。人工确认只 `interrupt()`，恢复时不重跑已经执行过的工具。",
        "服务在未同时传入 `llm_fn` 与 `tool_funcs` 时，用 LiteLLM、混合检索和进程内 MCP。`service.app:app` 第一次被访问时才 `create_knowdesk_app()`。",
        defaults_line,
        f"镜像把 tau2-bench 固定在 checkout `{checkout.group(1)}`。",
        "",
        "| 路径 | 入口 | 工具谁来执行 | 状态放在哪 |",
        "|---|---|---|---|",
        "| τ 评测 | `eval/tau_runner.py` | τ orchestrator | 适配器进程内的安全状态，图用 `InMemorySaver` |",
        "| 无框架循环 | 调用方自己建 `PlainAgentLoop` | `ReliableToolExecutor` | 调用方决定要不要 SQLite 账本 |",
        "| FastAPI 演示 | `service.app:app` | 进程内 MCP | `SqliteSaver` 加 `idempotency_ledger` |",
        "",
        "## 实测结果",
        "",
        "下面三张表是首页上的全部实验数字。原始 JSON 和完整报告走相对链接，不铺在这里。",
        "",
        "### X1 检索消融（test 67）",
        "",
        "超参只在 dev 口语查询上选定。这里只抄 test 67 的口语查询和原始工单说明，列是 Recall@10 与 MRR。",
        "",
        *x1_lines,
        "",
        "### X2 银行域端到端（dev 30）",
        "",
        *x2_lines,
        "",
        "### X9 中文零售与 X10 规则拦截",
        "",
        *x9_lines,
        "",
        *x10_lines,
        "",
        "## 工程要点",
        "",
        "- 面包屑混合检索：BM25 块与 Qwen 向量做静态 RRF，在线默认带意图加权；重排只留在消融里。",
        "- Fail-Closed 动态 Schema Guard：解锁后按当次 schema 校验参数，违规在重试上限后返回固定拒绝并去掉 tool calls。",
        "- LangGraph 把人工确认放在独立 `interrupt` 节点。确认恢复不会把同一写操作再执行一遍。",
        "- SQLite 幂等账本按会话、工具名和规范化参数做 SHA-256。业务失败不写入，进程重启后成功写仍可重放。",
        "- 工具传输用 MCP Python SDK 的 `Client(..., mode=\"legacy\")`。追踪用 OpenTelemetry SDK；只有设置了 OTLP 端点才把 span 送到 Jaeger。",
        "",
        "## 演示台工具",
        "",
        "在线模型只看到下面这些函数。隐藏写操作走 `call_discoverable_agent_tool`，不逐个放进 schema。",
        "",
        *_live_tool_lines(),
        "",
        "## Quick Start",
        "",
        "```bash",
        "uv run --no-sync pytest -v -W ignore",
        "uv run --no-sync uvicorn service.app:app --host 0.0.0.0 --port 8000",
        "```",
        "",
        "演示台是 [http://localhost:8000](http://localhost:8000)。需要 `DASHSCOPE_API_KEY`。单测不需要。",
        "",
        "```bash",
        "uv run --no-sync python eval/run_x2.py",
        "uv run --no-sync python eval/run_x9_retail_zh.py",
        "uv run --no-sync python eval/injection.py --table-a",
        "uv run --no-sync python eval/render_readme.py",
        "```",
        "",
        "X2 是 dev 30、三组同一个 max_steps，步数写在 `RESULTS_X2.md` 标题里。X9 是中文 retail 前 30 单的 DB Match。两条都会调用百炼。X1 的 test 67 报告已经在 `RESULTS_X1.md`，生成器不重跑它。",
        f"生成时的收集摘要：{collected_line}。解析结果 collected `{counts['collected']}`。这是收集数量，不是通过数量。",
        "",
        "## 评测边界与诚实局限（Honest Limitations）",
        "",
        f"- 冻结切分在 `eval/splits.json`：seed `{splits['seed']}`，总数 `{splits['total']}`，dev `{splits['dev']}`，test `{splits['test']}`，regression `{splits['regression']}`。首页数字只来自链接到的 `RESULTS_*.md`；文件写明未跑，表就空着。",
        "- Guard 的失败关闭是 `max_guard_retries = 2` 之后的固定拒绝。它不是对任意输入的拦截率。X10 表 A 是规则单测，Guard OFF 的违规率按定义是 1.0。",
        "- `service/app.py` 与 `adapters/tau2_adapter.py` 都不构造 `PlainAgentLoop`。在线检索不调用 `search_with_subagent_isolation`。",
        "- 跨度里的脱敏是 `security.dlp.redact_text`。没有写进上表的通过率、费用或 Cohen's κ，就是这次发布没有那次测量。",
        "",
        "## 结果文件",
        "",
        *other_lines,
        *manifest_lines,
        "",
        "重新生成：`uv run --no-sync python eval/render_readme.py`。",
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
